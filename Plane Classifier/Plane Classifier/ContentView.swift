//
//  ContentView.swift
//  Plane Classifier
//
//  Live camera feed → YOLO plane detection → MobileNetV4 classification, with a
//  bounding box, the identified model, and a time-aggregated confidence readout.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraController()
    /// Zoom captured when a pinch begins, so the gesture scales relative to it.
    @State private var pinchStartZoom: CGFloat?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            switch camera.access {
            case .authorized:
                cameraScene
            case .denied:
                CameraDeniedView()
            case .unknown:
                ProgressView()
                    .tint(.white)
            }

            if let message = camera.statusMessage {
                ErrorBanner(message: message)
            }
        }
        .onAppear { camera.start() }
        .onDisappear { camera.stop() }
        .statusBarHidden(true)
        .preferredColorScheme(.dark)
    }

    private var cameraScene: some View {
        ZStack(alignment: .bottom) {
            CameraPreviewView(previewLayer: camera.previewLayer)
                .ignoresSafeArea()
                .contentShape(Rectangle())
                .gesture(zoomGesture)

            DetectionOverlayView(state: camera.state, previewLayer: camera.previewLayer)
                .ignoresSafeArea()
                .allowsHitTesting(false)

            VStack(spacing: 12) {
                ZoomControlBar(camera: camera)
                AggregatePanel(state: camera.state)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 8)
        }
    }

    /// Pinch to zoom continuously across the full optical + digital range.
    private var zoomGesture: some Gesture {
        MagnifyGesture()
            .onChanged { value in
                let start = pinchStartZoom ?? camera.displayZoom
                if pinchStartZoom == nil { pinchStartZoom = start }
                camera.setDisplayZoom(start * value.magnification)
            }
            .onEnded { _ in pinchStartZoom = nil }
    }
}

private struct CameraDeniedView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "camera.metering.unknown")
                .font(.system(size: 52))
                .foregroundStyle(.secondary)
            Text("Camera access needed")
                .font(.title3.weight(.semibold))
            Text("Enable camera access in Settings to detect and classify aircraft.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            if let url = URL(string: UIApplication.openSettingsURLString) {
                Link("Open Settings", destination: url)
                    .font(.headline)
                    .padding(.top, 4)
            }
        }
        .padding(32)
    }
}

private struct ErrorBanner: View {
    let message: String
    var body: some View {
        VStack {
            Text(message)
                .font(.footnote.weight(.medium))
                .foregroundStyle(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(Color.red.opacity(0.9), in: Capsule())
                .padding(.top, 12)
            Spacer()
        }
    }
}

#Preview {
    ContentView()
}
