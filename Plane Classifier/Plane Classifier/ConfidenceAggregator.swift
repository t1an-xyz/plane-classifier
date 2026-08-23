//
//  ConfidenceAggregator.swift
//  Plane Classifier
//
//  Turns the classifier's noisy per-frame distribution into a stable, "time-aggregated"
//  confidence by keeping a time-decayed exponential moving average (EMA) of the class
//  probabilities.
//
//  Using an exponential decay in *time* (rather than a fixed frame count) keeps the
//  behaviour consistent regardless of frame rate, and lets the aggregated confidence
//  naturally fade toward zero when the plane leaves the frame (we keep decaying with a
//  zero signal). `tau` is the time constant: larger = smoother/slower to react.
//

import Foundation
import QuartzCore

final class ConfidenceAggregator {

    /// Time constant in seconds. Higher = smoother and more "aggregated".
    private let tau: Double
    private var ema: [String: Double] = [:]
    private var lastUpdate: CFTimeInterval?

    init(tau: Double = 1.5) {
        self.tau = tau
    }

    /// Fold a fresh per-frame distribution into the running average.
    func update(with scores: [ClassScore], at time: CFTimeInterval) {
        let decay = decayFactor(to: time)
        let weight = 1.0 - decay

        // Decay every class we already track...
        for key in ema.keys {
            ema[key]! *= decay
        }
        // ...then blend in the newest observation.
        for score in scores {
            ema[score.label, default: 0] += weight * Double(score.confidence)
        }
    }

    /// No plane this frame: keep decaying toward zero so stale confidence fades out.
    func decay(at time: CFTimeInterval) {
        let decay = decayFactor(to: time)
        for key in ema.keys {
            ema[key]! *= decay
        }
    }

    /// Reset all accumulated state.
    func reset() {
        ema.removeAll()
        lastUpdate = nil
    }

    /// The current time-aggregated distribution, sorted by confidence descending.
    func topClasses(limit: Int) -> [ClassScore] {
        ema.map { ClassScore(label: $0.key, confidence: Float($0.value)) }
            .sorted { $0.confidence > $1.confidence }
            .prefix(limit)
            .map { $0 }
    }

    /// `exp(-dt / tau)`; on the very first sample this is 0 so the EMA adopts it outright.
    private func decayFactor(to time: CFTimeInterval) -> Double {
        defer { lastUpdate = time }
        guard let last = lastUpdate else { return 0 }
        let dt = max(0, time - last)
        return exp(-dt / tau)
    }
}
