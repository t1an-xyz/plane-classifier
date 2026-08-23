//
//  DetectionOverlayView.swift
//  Plane Classifier
//
//  Draws the bounding box + instantaneous label over the camera preview, and the
//  bottom panel showing the time-aggregated model guess, its aggregated confidence,
//  and the top-3 aggregated classes as bars.
//

import SwiftUI
import AVFoundation

// MARK: - Bounding box + instant label

struct DetectionOverlayView: View {
    let state: DetectionState
    let previewLayer: AVCaptureVideoPreviewLayer

    var body: some View {
        ZStack(alignment: .topLeading) {
            if let rect = boxRect() {
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(Color.green, lineWidth: 3)
                    .frame(width: rect.width, height: rect.height)
                    .offset(x: rect.minX, y: rect.minY)

                labelChip
                    .fixedSize()
                    .offset(x: rect.minX, y: labelY(for: rect))
            }
        }
        // Fill the whole preview and pin to top-leading so `.offset` coordinates
        // (computed in `previewLayer.bounds` top-left space by `boxRect()`) map
        // correctly. Without this, the ZStack collapses to the box's size, which
        // both mis-positions the box and resizes the shared layout every frame —
        // jittering the sibling AggregatePanel.
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .allowsHitTesting(false)
        .animation(.easeOut(duration: 0.15), value: state.box)
    }

    private var labelChip: some View {
        Text(instantText)
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color.green.opacity(0.85), in: Capsule())
    }

    private var instantText: String {
        guard let instant = state.instant else { return "Identifying…" }
        return "\(instant.label) · \(percent(instant.confidence))"
    }

    /// Map a Vision-normalized box (origin bottom-left) onto the preview.
    ///
    /// We do the `resizeAspectFill` transform by hand instead of using
    /// `AVCaptureVideoPreviewLayer.layerRectConverted(fromMetadataOutputRect:)` because
    /// that helper is zoom-aware: it expects pre-zoom sensor coordinates, whereas Vision
    /// runs on the already-zoomed video-output buffer. Doing the mapping ourselves keeps
    /// the box correct at any zoom / active lens, since the preview and Vision share the
    /// exact same (zoomed) frame — only the aspect-fill fit matters.
    private func boxRect() -> CGRect? {
        guard let box = state.box, let frame = state.frameSize else { return nil }
        let bounds = previewLayer.bounds
        guard bounds.width > 1, bounds.height > 1, frame.width > 1, frame.height > 1 else { return nil }

        let scale = max(bounds.width / frame.width, bounds.height / frame.height)
        let scaledW = frame.width * scale
        let scaledH = frame.height * scale
        let originX = (bounds.width - scaledW) / 2
        let originY = (bounds.height - scaledH) / 2

        // Vision's origin is bottom-left; flip Y to the layer's top-left space.
        let rect = CGRect(
            x: originX + box.minX * scaledW,
            y: originY + (1 - box.maxY) * scaledH,
            width: box.width * scaledW,
            height: box.height * scaledH
        )
        guard !rect.isNull, !rect.isInfinite, rect.width > 1, rect.height > 1 else { return nil }
        return rect
    }

    private func labelY(for rect: CGRect) -> CGFloat {
        rect.minY > 30 ? rect.minY - 30 : rect.minY + 4
    }
}

// MARK: - Lens / zoom picker

/// A compact lens picker styled after the system Camera app. Tapping a stop ramps the
/// zoom to that lens — including the telephoto ("telescopic") lens on capable devices.
struct ZoomControlBar: View {
    @ObservedObject var camera: CameraController

    var body: some View {
        if camera.lensPresets.count > 1 {
            HStack(spacing: 6) {
                ForEach(camera.lensPresets) { preset in
                    Button {
                        camera.select(preset)
                    } label: {
                        Text(isActive(preset) ? "\(currentLabel)×" : preset.label)
                            .font(.footnote.weight(.semibold))
                            .monospacedDigit()
                            .foregroundStyle(isActive(preset) ? Color.yellow : .white)
                            .frame(minWidth: isActive(preset) ? 40 : 30, minHeight: 34)
                            .background(
                                Circle()
                                    .fill(.white.opacity(isActive(preset) ? 0.18 : 0.001))
                                    .frame(width: 34, height: 34),
                                alignment: .center
                            )
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(
                Capsule().fill(Color.black.opacity(0.45))
                    .overlay(Capsule().strokeBorder(.white.opacity(0.12), lineWidth: 1))
            )
            .animation(.easeOut(duration: 0.15), value: camera.displayZoom)
        }
    }

    /// The label shown on the active stop: the live zoom value (e.g. "2.3").
    private var currentLabel: String {
        let x = camera.displayZoom
        if x < 1 { return String(format: "%.1f", x) }
        if abs(x - x.rounded()) < 0.05 { return String(Int(x.rounded())) }
        return String(format: "%.1f", x)
    }

    /// The active stop is the highest one at or below the current zoom.
    private func isActive(_ preset: LensPreset) -> Bool {
        let active = camera.lensPresets.last { $0.displayX <= camera.displayZoom + 0.05 }
            ?? camera.lensPresets.first
        return preset.id == active?.id
    }
}

// MARK: - Aggregated readout panel

struct AggregatePanel: View {
    let state: DetectionState

    private var locked: Bool { (state.aggregatedTop?.confidence ?? 0) > 0.4 }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header

            VStack(spacing: 8) {
                ForEach(state.aggregated) { score in
                    ConfidenceBar(score: score,
                                  isTop: score.id == state.aggregatedTop?.id)
                }
            }

            Text("Point the camera at an aircraft.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .opacity(state.aggregated.isEmpty ? 1 : 0)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(Color.black.opacity(0.55))
                .overlay(
                    RoundedRectangle(cornerRadius: 20)
                        .strokeBorder(.white.opacity(0.12), lineWidth: 1)
                )
        )
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text(state.hasPlane ? "AIRCRAFT DETECTED" : "SCANNING")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(state.hasPlane ? Color.green : Color.secondary)
                Text(locked ? (state.aggregatedTop?.label ?? "—") : "Searching…")
                    .font(.title2.weight(.bold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.6)
            }
            Spacer()
            if let top = state.aggregatedTop {
                VStack(alignment: .trailing, spacing: 0) {
                    Text(percent(top.confidence))
                        .font(.system(size: 34, weight: .heavy, design: .rounded))
                        .monospacedDigit()
                        .foregroundStyle(locked ? Color.green : Color.primary)
                    Text("time-aggregated")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
}

// MARK: - One confidence bar

struct ConfidenceBar: View {
    let score: ClassScore
    let isTop: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(score.label)
                    .font(.subheadline.weight(isTop ? .semibold : .regular))
                    .lineLimit(1)
                Spacer()
                Text(percent(score.confidence))
                    .font(.subheadline.weight(.semibold))
                    .monospacedDigit()
                    .foregroundStyle(isTop ? Color.green : .secondary)
            }
            // Proportional bar without GeometryReader (which is fragile inside a
            // ForEach on this OS): the fill spans the full track then scales in X.
            Capsule()
                .fill(.white.opacity(0.12))
                .frame(height: 6)
                .overlay(alignment: .leading) {
                    Capsule()
                        .fill(isTop ? Color.green : Color.white.opacity(0.5))
                        .scaleEffect(x: fillFraction, y: 1, anchor: .leading)
                }
        }
    }

    private var fillFraction: CGFloat {
        CGFloat(max(0.001, min(1, score.confidence)))
    }
}

// MARK: - Shared formatting

func percent(_ value: Float) -> String {
    "\(Int((max(0, min(1, value)) * 100).rounded()))%"
}
