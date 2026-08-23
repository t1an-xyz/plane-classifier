//
//  PlaneVisionPipeline.swift
//  Plane Classifier
//
//  Two-stage on-device vision pipeline:
//    1. YOLOv8n (CoreML, COCO) detects aircraft and returns the largest "airplane" box.
//    2. The box is padded, cropped from the frame, and fed to the MobileNetV4 classifier
//       (PlaneClassifier.mlpackage) which returns a calibrated distribution over the 16
//       commercial jet families.
//
//  This type is deliberately `nonisolated` so it can run on the camera's background
//  video queue, away from the main actor. It is `@unchecked Sendable` because it holds
//  only immutable, thread-safe `VNCoreMLModel` references and builds a fresh request +
//  handler per frame (Vision models are safe to use concurrently this way).
//

import Foundation
import Vision
import CoreML
import CoreImage
import CoreVideo
import ImageIO

/// One frame's worth of pipeline output. `Sendable` so it can cross from the
/// video queue back to the main actor.
struct ClassScore: Identifiable, Sendable, Equatable {
    let label: String
    let confidence: Float
    var id: String { label }
}

struct PipelineOutput: Sendable {
    /// Detection box for the plane, Vision-normalized (0...1, origin bottom-left).
    let box: CGRect
    /// Full classifier distribution, sorted by confidence descending.
    let probabilities: [ClassScore]
    /// Size of the processed frame (in the upright orientation Vision saw it). Used by
    /// the overlay to map the normalized box onto the aspect-fill preview.
    let frameSize: CGSize

    var top: ClassScore? { probabilities.first }
}

nonisolated final class PlaneVisionPipeline: @unchecked Sendable {

    enum PipelineError: LocalizedError {
        case missingModel(String)
        var errorDescription: String? {
            switch self {
            case .missingModel(let name):
                return "Couldn't find the compiled \(name) model in the app bundle."
            }
        }
    }

    /// The COCO label the detector must match for us to treat a box as an aircraft.
    private static let aircraftLabel = "airplane"
    /// Padding added around the detected box (fraction of box size per side) before
    /// cropping for classification. Matches the notebook's BBOX_PADDING (8%).
    private static let boxPadding: CGFloat = 0.08

    private let detector: VNCoreMLModel
    private let classifier: VNCoreMLModel

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all // let CoreML place work on the Neural Engine when possible

        guard let detURL = Bundle.main.url(forResource: "yolov8n", withExtension: "mlmodelc") else {
            throw PipelineError.missingModel("yolov8n")
        }
        guard let clsURL = Bundle.main.url(forResource: "PlaneClassifier", withExtension: "mlmodelc") else {
            throw PipelineError.missingModel("PlaneClassifier")
        }

        detector = try VNCoreMLModel(for: MLModel(contentsOf: detURL, configuration: config))
        classifier = try VNCoreMLModel(for: MLModel(contentsOf: clsURL, configuration: config))
    }

    /// Run detection + classification on a single frame.
    /// - Parameter orientation: how to interpret `pixelBuffer` as upright. The camera
    ///   controller rotates its output to portrait, so this is `.up` in normal use.
    /// - Returns: the largest airplane's box + its classification, or `nil` if no plane.
    func process(pixelBuffer: CVPixelBuffer,
                 orientation: CGImagePropertyOrientation = .up) -> PipelineOutput? {

        // Frame size as Vision sees it once `orientation` is applied. The detection box
        // is normalized against this, so the overlay uses it for the aspect-fill mapping.
        let bufferW = CVPixelBufferGetWidth(pixelBuffer)
        let bufferH = CVPixelBufferGetHeight(pixelBuffer)
        let frameSize: CGSize
        switch orientation {
        case .left, .right, .leftMirrored, .rightMirrored:
            frameSize = CGSize(width: bufferH, height: bufferW) // orientation swaps axes
        default:
            frameSize = CGSize(width: bufferW, height: bufferH)
        }

        // ---- Stage 1: YOLO detection -------------------------------------------------
        let detRequest = VNCoreMLRequest(model: detector)
        detRequest.imageCropAndScaleOption = .scaleFill // detect over the whole frame

        let detHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                               orientation: orientation,
                                               options: [:])
        do {
            try detHandler.perform([detRequest])
        } catch {
            return nil
        }

        guard let observations = detRequest.results as? [VNRecognizedObjectObservation] else {
            return nil
        }

        // Keep only aircraft, then take the largest (matches the notebook's crop logic).
        let planes = observations.filter { $0.labels.first?.identifier == Self.aircraftLabel }
        guard let best = planes.max(by: { Self.area($0.boundingBox) < Self.area($1.boundingBox) }) else {
            return nil
        }
        let detectionBox = best.boundingBox

        // ---- Stage 2: crop + classify ------------------------------------------------
        let unit = CGRect(x: 0, y: 0, width: 1, height: 1)
        let padded = detectionBox
            .insetBy(dx: -detectionBox.width * Self.boxPadding,
                     dy: -detectionBox.height * Self.boxPadding)
            .intersection(unit)

        // CIImage shares Vision's coordinate convention (origin bottom-left), so the
        // normalized box maps straight onto the oriented image extent.
        let oriented = CIImage(cvPixelBuffer: pixelBuffer).oriented(orientation)
        let extent = oriented.extent
        let cropRect = CGRect(
            x: extent.minX + padded.minX * extent.width,
            y: extent.minY + padded.minY * extent.height,
            width: padded.width * extent.width,
            height: padded.height * extent.height
        ).integral

        guard !cropRect.isNull, cropRect.width >= 8, cropRect.height >= 8 else {
            // Detected a plane but the crop is degenerate; report the box without a label.
            return PipelineOutput(box: detectionBox, probabilities: [], frameSize: frameSize)
        }

        let crop = oriented
            .cropped(to: cropRect)
            .transformed(by: CGAffineTransform(translationX: -cropRect.minX, y: -cropRect.minY))

        let clsRequest = VNCoreMLRequest(model: classifier)
        clsRequest.imageCropAndScaleOption = .scaleFit // letterbox to 256x256 like eval training

        let clsHandler = VNImageRequestHandler(ciImage: crop, options: [:])
        do {
            try clsHandler.perform([clsRequest])
        } catch {
            return PipelineOutput(box: detectionBox, probabilities: [], frameSize: frameSize)
        }

        guard let classifications = clsRequest.results as? [VNClassificationObservation] else {
            return PipelineOutput(box: detectionBox, probabilities: [], frameSize: frameSize)
        }

        let scores = classifications.map { ClassScore(label: $0.identifier, confidence: $0.confidence) }
        return PipelineOutput(box: detectionBox, probabilities: scores, frameSize: frameSize)
    }

    private static func area(_ rect: CGRect) -> CGFloat { rect.width * rect.height }
}
