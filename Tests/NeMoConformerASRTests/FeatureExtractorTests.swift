import XCTest
import NeMoFeatureExtractor
@testable import NeMoConformerASR

final class FeatureExtractorTests: XCTestCase {

    /// Test feature extractor with sine wave against Python reference
    func testFeatureExtractorMatchesPythonReference() throws {
        // Load test sine wave (160,000 samples = 10 seconds at 16kHz)
        let sineWaveURL = Bundle.module.url(forResource: "test_sine_wave", withExtension: "bin")!
        let sineWaveData = try Data(contentsOf: sineWaveURL)
        let sineWave = sineWaveData.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float.self))
        }

        // Load reference mel spectrogram from Python (shape: 1, 80, 1001 -> flattened)
        let melRefURL = Bundle.module.url(forResource: "test_mel_reference", withExtension: "bin")!
        let melRefData = try Data(contentsOf: melRefURL)
        let melReference = melRefData.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float.self))
        }

        // Load metadata
        let metadataURL = Bundle.module.url(forResource: "test_metadata", withExtension: "json")!
        let metadataData = try Data(contentsOf: metadataURL)
        let metadata = try JSONDecoder().decode(TestMetadata.self, from: metadataData)

        // Verify input dimensions
        XCTAssertEqual(sineWave.count, metadata.numSamples,
                       "Sine wave should have \(metadata.numSamples) samples")

        // Expected mel shape: (1, 80, 1001) -> flattened = 80080
        let expectedMelCount = metadata.melShape[1] * metadata.melShape[2]
        XCTAssertEqual(melReference.count, expectedMelCount,
                       "Reference mel should have \(expectedMelCount) values, got \(melReference.count)")

        // Initialize feature extractor with NeMo ASR config
        let featureExtractor = NeMoFeatureExtractor(config: .nemoASR)

        // Process sine wave - returns [[Float]] with shape [nMels][nFrames]
        let melOutput = try featureExtractor.process(samples: sineWave)

        // Verify output dimensions
        let outputNMels = melOutput.count
        let outputNFrames = melOutput.first?.count ?? 0

        XCTAssertEqual(outputNMels, metadata.nMels,
                       "nMels should be \(metadata.nMels), got \(outputNMels)")
        XCTAssertEqual(outputNFrames, metadata.melFrames,
                       "Frames should be \(metadata.melFrames), got \(outputNFrames)")

        print("Input: \(sineWave.count) samples (\(metadata.durationSec) sec)")
        print("Output shape: [\(outputNMels), \(outputNFrames)]")
        print("Expected shape: [\(metadata.nMels), \(metadata.melFrames)]")

        // Compare each frame
        // Reference is stored as [1, 80, 1001] row-major (C-style):
        //   index = batch * (80 * 1001) + mel * 1001 + frame
        // Since batch=0, index = mel * 1001 + frame
        // Our output is [[Float]] with [mel][frame] access

        let nMels = metadata.nMels
        let nFrames = metadata.melFrames

        var maxDiff: Float = 0
        var sumDiff: Float = 0
        var diffCount = 0

        for mel in 0..<nMels {
            for frame in 0..<nFrames {
                let refIdx = mel * nFrames + frame
                let refValue = melReference[refIdx]
                let outValue = melOutput[mel][frame]

                let diff = abs(refValue - outValue)
                maxDiff = max(maxDiff, diff)
                sumDiff += diff
                diffCount += 1
            }
        }

        let avgDiff = sumDiff / Float(diffCount)

        print("\nComparison results:")
        print("  Max difference: \(maxDiff)")
        print("  Avg difference: \(avgDiff)")
        print("  Total values compared: \(diffCount)")

        // Sample some values for debugging
        print("\nSample values (mel=0, frames 0-4):")
        for frame in 0..<min(5, nFrames) {
            let refIdx = 0 * nFrames + frame
            print("  frame \(frame): ref=\(melReference[refIdx]), out=\(melOutput[0][frame])")
        }

        // Assert acceptable tolerance
        // Note: Some difference is expected due to float precision and implementation details
        XCTAssertLessThan(maxDiff, 1.0,
                          "Max difference should be less than 1.0, got \(maxDiff)")
        XCTAssertLessThan(avgDiff, 0.1,
                          "Average difference should be less than 0.1, got \(avgDiff)")
    }
}

// MARK: - Test Metadata

private struct TestMetadata: Decodable {
    let sampleRate: Int
    let numSamples: Int
    let durationSec: Int
    let frequencyHz: Int
    let melShape: [Int]
    let melFrames: Int
    let nMels: Int

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case numSamples = "num_samples"
        case durationSec = "duration_sec"
        case frequencyHz = "frequency_hz"
        case melShape = "mel_shape"
        case melFrames = "mel_frames"
        case nMels = "n_mels"
    }
}
