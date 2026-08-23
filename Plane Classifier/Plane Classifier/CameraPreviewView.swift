//
//  CameraPreviewView.swift
//  Plane Classifier
//
//  Bridges the AVCaptureVideoPreviewLayer into SwiftUI. The layer fills the same bounds
//  as the SwiftUI overlay, so `AVCaptureVideoPreviewLayer.layerRectConverted(...)` gives
//  detection-box coordinates that are directly usable in the overlay.
//

import SwiftUI
import AVFoundation

struct CameraPreviewView: UIViewRepresentable {
    let previewLayer: AVCaptureVideoPreviewLayer

    func makeUIView(context: Context) -> PreviewContainerView {
        let view = PreviewContainerView()
        view.backgroundColor = .black
        view.attach(previewLayer)
        return view
    }

    func updateUIView(_ uiView: PreviewContainerView, context: Context) {}
}

/// A plain UIView that keeps the capture preview layer sized to its bounds.
final class PreviewContainerView: UIView {
    private weak var previewLayer: AVCaptureVideoPreviewLayer?

    func attach(_ layer: AVCaptureVideoPreviewLayer) {
        previewLayer = layer
        layer.frame = bounds
        self.layer.addSublayer(layer)
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer?.frame = bounds
    }
}
