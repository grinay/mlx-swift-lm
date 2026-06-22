import Foundation
import Testing

@testable import MLXVLM

/// Regression tests for the prompt-prefix KV cache boundary logic
/// (`Mistral3VLM.recordBoundary`). The record path used to compute a boundary
/// that could equal the full token count `n`, leaving an empty suffix slice
/// `[boundary..<n]`. Prefilling that zero-length sequence fed an empty array
/// into attention and crashed MLX's `reshape` ("Cannot infer the shape of an
/// empty array"), taking down the inference helper. Reported as Sentry
/// RECALL-ADHD-3A. The boundary math is extracted as a pure function so the
/// guard is testable without loading model weights.
struct Mistral3PrefixCacheTests {

    /// Build a token run [0, 1, 2, ...] of the given length.
    private func tokens(_ count: Int) -> [Int32] {
        (0 ..< count).map { Int32($0) }
    }

    @Test("Identical text-only request yields no suffix → skip (RECALL-ADHD-3A)")
    func identicalTextOnlyRequestSkips() {
        // Two consecutive identical text-only requests: flat == lastTokens,
        // no image token so firstImage == n. lcp == n → boundary would be n →
        // empty suffix. Must skip, not record.
        let flat = tokens(300)
        let boundary = Mistral3VLM.recordBoundary(
            flat: flat, lastTokens: flat, firstImage: flat.count)
        #expect(boundary == nil)
    }

    @Test("Current request is a strict prefix of the previous → skip")
    func currentIsPrefixOfPreviousSkips() {
        // Shorter text-only request whose whole sequence is a prefix of the
        // previous one: lcp == n, firstImage == n → boundary == n → empty
        // suffix. Must skip.
        let prev = tokens(300)
        let flat = Array(prev[0 ..< 200])
        let boundary = Mistral3VLM.recordBoundary(
            flat: flat, lastTokens: prev, firstImage: flat.count)
        #expect(boundary == nil)
    }

    @Test("Shared prefix then divergence (text-only) → records at the LCP")
    func divergingTextOnlyRecordsAtLCP() {
        // 200 shared tokens, then the requests diverge. Non-empty suffix.
        var prev = tokens(200)
        prev += (0 ..< 100).map { Int32(1000 + $0) }
        var flat = tokens(200)
        flat += (0 ..< 100).map { Int32(2000 + $0) }
        let boundary = Mistral3VLM.recordBoundary(
            flat: flat, lastTokens: prev, firstImage: flat.count)
        #expect(boundary == 200)
    }

    @Test("Boundary is capped at the first image token")
    func boundaryCappedAtFirstImage() {
        // LCP is long (250) but the image token sits at 150 — image features
        // are per-request, never cacheable, so the boundary caps at 150.
        // 150 < n leaves a non-empty suffix → record.
        let flat = tokens(400)
        let boundary = Mistral3VLM.recordBoundary(
            flat: flat, lastTokens: flat, firstImage: 150)
        #expect(boundary == 150)
    }

    @Test("Prefix shorter than the minimum is not worth caching → skip")
    func shortPrefixSkips() {
        var prev = tokens(50)
        prev += (0 ..< 100).map { Int32(1000 + $0) }
        var flat = tokens(50)
        flat += (0 ..< 100).map { Int32(2000 + $0) }
        let boundary = Mistral3VLM.recordBoundary(
            flat: flat, lastTokens: prev, firstImage: flat.count)
        #expect(boundary == nil)
    }
}
