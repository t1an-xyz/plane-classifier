//
//  CameraController.swift
//  Plane Classifier
//
//  Owns the AVCaptureSession, streams frames to the vision pipeline on a background
//  queue, folds the results into the time-aggregated confidence, and publishes a
//  `DetectionState` for the UI.
//

import Foundation
import AVFoundation
import SwiftUI
import Combine
import ImageIO

/// Everything the UI needs to render one moment in time. `Sendable` so it can be
/// produced off the main actor and handed back safely.
struct DetectionState: Sendable, Equatable {
    /// Detected plane box, Vision-normalized (origin bottom-left). `nil` when no plane.
    var box: CGRect?
    /// Size of the frame the box is normalized against (for the overlay's mapping).
    var frameSize: CGSize?
    /// This frame's single best guess (before time-aggregation).
    var instant: ClassScore?
    /// Time-aggregated distribution (top-N), most confident first.
    var aggregated: [ClassScore]
    /// Whether a plane was detected on the latest processed frame.
    var hasPlane: Bool

    /// The time-aggregated best guess (what the big readout shows).
    var aggregatedTop: ClassScore? { aggregated.first }

    static let empty = DetectionState(box: nil, frameSize: nil, instant: nil, aggregated: [], hasPlane: false)
}

// MARK: - Zoom / lens model

/// A tappable zoom stop shown in the lens picker. Optical stops correspond to a real
/// physical lens on the device (e.g. the telephoto / "telescopic" lens); the system
/// engages that lens automatically when the device zoom factor reaches its base.
struct LensPreset: Identifiable, Equatable {
    let id: String
    /// Short label, e.g. "0.5", "1", "3".
    let label: String
    /// The zoom multiplier relative to the wide (1×) lens, e.g. 0.5, 1, 3.
    let displayX: CGFloat
    /// The raw `videoZoomFactor` to set on the active device to reach this stop.
    let deviceZoomFactor: CGFloat
    /// Whether this stop maps to a dedicated optical lens (vs. a digital-zoom stop).
    let isOptical: Bool
}

// MARK: - Frame handler

/// Bridges AVFoundation's background sample-buffer callback to the vision pipeline.
/// `nonisolated` so it lives entirely off the main actor.
nonisolated final class FrameHandler: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    private let pipeline: PlaneVisionPipeline
    private let onOutput: @Sendable (PipelineOutput?) -> Void

    init(pipeline: PlaneVisionPipeline, onOutput: @escaping @Sendable (PipelineOutput?) -> Void) {
        self.pipeline = pipeline
        self.onOutput = onOutput
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        // The capture connection is rotated to portrait, so the buffer is already upright.
        let result = pipeline.process(pixelBuffer: pixelBuffer, orientation: .up)
        onOutput(result)
    }
}

// MARK: - Camera controller

@MainActor
final class CameraController: NSObject, ObservableObject {

    enum Access { case unknown, authorized, denied }

    @Published private(set) var access: Access = .unknown
    @Published private(set) var state: DetectionState = .empty
    @Published private(set) var statusMessage: String?

    /// Zoom stops for the available lenses, ascending (e.g. 0.5×, 1×, 3×).
    @Published private(set) var lensPresets: [LensPreset] = []
    /// Current zoom expressed as a multiplier of the wide (1×) lens.
    @Published private(set) var displayZoom: CGFloat = 1
    /// Bounds for the pinch gesture, in the same 1×-relative units as `displayZoom`.
    @Published private(set) var minDisplayZoom: CGFloat = 1
    @Published private(set) var maxDisplayZoom: CGFloat = 1

    /// Displayed by `CameraPreviewView` and used to map detection boxes to screen space.
    let previewLayer = AVCaptureVideoPreviewLayer()

    // AVCaptureSession is internally thread-safe; we drive it from a dedicated queue.
    nonisolated(unsafe) private let session = AVCaptureSession()
    nonisolated(unsafe) private let videoOutput = AVCaptureVideoDataOutput()
    nonisolated private let sessionQueue = DispatchQueue(label: "camera.session")
    nonisolated private let videoQueue = DispatchQueue(label: "camera.video.output",
                                                       qos: .userInitiated)
    nonisolated(unsafe) private var isConfigured = false
    nonisolated(unsafe) private var frameHandler: FrameHandler?
    /// The camera currently feeding the session. Configuration runs on `sessionQueue`,
    /// and zoom changes lock this device before mutating `videoZoomFactor`.
    nonisolated(unsafe) private var activeDevice: AVCaptureDevice?
    /// The `videoZoomFactor` that corresponds to the "1×" wide-angle framing.
    private var wideBaseZoom: CGFloat = 1

    private var pipeline: PlaneVisionPipeline?
    private let aggregator = ConfidenceAggregator()

    override init() {
        super.init()
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.session = session
        do {
            pipeline = try PlaneVisionPipeline()
        } catch {
            statusMessage = error.localizedDescription
        }
    }

    // MARK: Lifecycle

    func start() {
        guard pipeline != nil else { return } // model failed to load; statusMessage set
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            access = .authorized
            run()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                Task { @MainActor in
                    self.access = granted ? .authorized : .denied
                    if granted { self.run() }
                }
            }
        default:
            access = .denied
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    // MARK: Internals

    private func run() {
        guard let pipeline else { return }

        if frameHandler == nil {
            frameHandler = FrameHandler(pipeline: pipeline) { [weak self] output in
                // Hop back to the main actor to update published state.
                Task { @MainActor [weak self] in self?.ingest(output) }
            }
        }

        sessionQueue.async {
            self.configureSessionIfNeeded()
            if !self.session.isRunning { self.session.startRunning() }
            Task { @MainActor in self.applyPreviewRotation() }
        }
    }

    nonisolated private func configureSessionIfNeeded() {
        guard !isConfigured else { return }
        isConfigured = true

        session.beginConfiguration()
        session.sessionPreset = .high

        if let device = Self.bestBackCamera(),
           let input = try? AVCaptureDeviceInput(device: device),
           session.canAddInput(input) {
            session.addInput(input)
            activeDevice = device
            configureZoom(for: device)
        }

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true // drop frames while we're busy
        videoOutput.setSampleBufferDelegate(frameHandler, queue: videoQueue)
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }

        // Deliver buffers already rotated to portrait so Vision can treat them as upright.
        if let connection = videoOutput.connection(with: .video),
           connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }

        session.commitConfiguration()
    }

    // MARK: Lens selection & zoom

    /// Prefer a virtual multi-camera device so we can zoom smoothly across the physical
    /// lenses (ultra-wide / wide / telephoto). The system swaps to the telephoto
    /// ("telescopic") lens automatically once the zoom factor reaches its range.
    nonisolated private static func bestBackCamera() -> AVCaptureDevice? {
        let preferred: [AVCaptureDevice.DeviceType] = [
            .builtInTripleCamera,   // ultra-wide + wide + telephoto
            .builtInDualCamera,     // wide + telephoto
            .builtInDualWideCamera, // ultra-wide + wide
            .builtInWideAngleCamera // wide only
        ]
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: preferred, mediaType: .video, position: .back)
        for type in preferred {
            if let device = discovery.devices.first(where: { $0.deviceType == type }) {
                return device
            }
        }
        return AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
    }

    /// Derive the lens presets + zoom bounds from a device and publish them, then open
    /// at the wide (1×) framing. Runs on `sessionQueue` inside session configuration.
    nonisolated private func configureZoom(for device: AVCaptureDevice) {
        let minZoom = device.minAvailableVideoZoomFactor
        let maxZoom = device.maxAvailableVideoZoomFactor

        // Base `videoZoomFactor` for each physical lens on the device.
        let constituents = device.constituentDevices
        let switchOvers = device.virtualDeviceSwitchOverVideoZoomFactors.map { CGFloat(truncating: $0) }
        var lensBases: [(type: AVCaptureDevice.DeviceType, base: CGFloat)] = []
        if constituents.isEmpty {
            lensBases = [(device.deviceType, minZoom)]
        } else {
            for (i, camera) in constituents.enumerated() {
                let base = (i == 0) ? minZoom : (switchOvers[safe: i - 1] ?? minZoom)
                lensBases.append((camera.deviceType, base))
            }
        }

        // "1×" is the wide-angle lens; everything else is expressed relative to it.
        let wideBase = lensBases.first(where: { $0.type == .builtInWideAngleCamera })?.base ?? minZoom
        let maxX = min(maxZoom / wideBase, 15) // cap runaway digital zoom
        let minX = minZoom / wideBase

        var presets: [LensPreset] = lensBases.map { entry in
            let x = entry.base / wideBase
            return LensPreset(id: "optical-\(entry.type.rawValue)",
                              label: Self.zoomLabel(x),
                              displayX: x,
                              deviceZoomFactor: entry.base,
                              isOptical: true)
        }

        // Wide-only devices get a couple of digital-zoom stops so the picker is useful
        // for spotting distant aircraft.
        if presets.count <= 1 {
            for x in [CGFloat(2), 4] where x <= maxX {
                presets.append(LensPreset(id: "digital-\(x)",
                                          label: Self.zoomLabel(x),
                                          displayX: x,
                                          deviceZoomFactor: x * wideBase,
                                          isOptical: false))
            }
        }
        presets.sort { $0.displayX < $1.displayX }
        let finalPresets = presets

        // Open at 1× wide (a triple/dual-wide device otherwise starts at ultra-wide).
        let initialFactor = max(min(wideBase, maxZoom), minZoom)
        if (try? device.lockForConfiguration()) != nil {
            device.videoZoomFactor = initialFactor
            device.unlockForConfiguration()
        }

        Task { @MainActor in
            self.wideBaseZoom = wideBase
            self.minDisplayZoom = minX
            self.maxDisplayZoom = maxX
            self.lensPresets = finalPresets
            self.displayZoom = initialFactor / wideBase
        }
    }

    /// Zoom to a stop (used by the lens picker), ramping for a smooth transition.
    func select(_ preset: LensPreset) {
        setDisplayZoom(preset.displayX, animated: true)
    }

    /// Set the zoom, expressed as a multiplier of the wide (1×) lens. On a multi-camera
    /// device the system engages the matching physical lens (incl. telephoto) for us.
    func setDisplayZoom(_ x: CGFloat, animated: Bool = false) {
        let clampedX = min(max(x, minDisplayZoom), maxDisplayZoom)
        displayZoom = clampedX
        let factor = clampedX * wideBaseZoom
        sessionQueue.async {
            guard let device = self.activeDevice,
                  (try? device.lockForConfiguration()) != nil else { return }
            let target = min(max(factor, device.minAvailableVideoZoomFactor),
                             device.maxAvailableVideoZoomFactor)
            if animated {
                device.ramp(toVideoZoomFactor: target, withRate: 8)
            } else {
                device.cancelVideoZoomRamp()
                device.videoZoomFactor = target
            }
            device.unlockForConfiguration()
        }
    }

    nonisolated private static func zoomLabel(_ x: CGFloat) -> String {
        if x < 1 { return String(format: "%.1f", x) }        // 0.5
        if abs(x - x.rounded()) < 0.05 { return String(Int(x.rounded())) } // 1, 3
        return String(format: "%.1f", x)                     // 2.5
    }

    private func applyPreviewRotation() {
        if let connection = previewLayer.connection,
           connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }
    }

    /// Runs on the main actor: fold the frame result into the time-aggregated confidence.
    private func ingest(_ output: PipelineOutput?) {
        let now = CACurrentMediaTime()
        if let output, !output.probabilities.isEmpty {
            aggregator.update(with: output.probabilities, at: now)
        } else {
            aggregator.decay(at: now)
        }

        state = DetectionState(
            box: output?.box,
            frameSize: output?.frameSize,
            instant: output?.top,
            aggregated: aggregator.topClasses(limit: 3),
            hasPlane: output != nil
        )
    }
}

private extension Array {
    /// Bounds-checked subscript; returns `nil` instead of trapping on a bad index.
    nonisolated subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
