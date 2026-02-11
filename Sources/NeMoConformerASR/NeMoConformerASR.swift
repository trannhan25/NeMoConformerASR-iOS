import Foundation
import CoreML
import NeMoFeatureExtractor

/// NeMo Conformer ASR errors
public enum NeMoConformerASRError: Error, LocalizedError {
    case modelLoadFailed(String)
    case vocabularyLoadFailed(String)
    case invalidInput(String)
    case inferenceFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let message):
            return "Model load failed: \(message)"
        case .vocabularyLoadFailed(let message):
            return "Vocabulary load failed: \(message)"
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        case .inferenceFailed(let message):
            return "Inference failed: \(message)"
        }
    }
}

/// Supported input durations for the Conformer model
public enum ConformerInputDuration: CaseIterable {
    case fiveSeconds     // 5 sec = 80,000 samples → 501 mel → 126 encoded
    case tenSeconds      // 10 sec = 160,000 samples → 1001 mel → 251 encoded
    case fifteenSeconds  // 15 sec = 240,000 samples → 1501 mel → 376 encoded
    case twentySeconds   // 20 sec = 320,000 samples → 2001 mel → 501 encoded

    public var samples: Int {
        switch self {
        case .fiveSeconds: return 80_000
        case .tenSeconds: return 160_000
        case .fifteenSeconds: return 240_000
        case .twentySeconds: return 320_000
        }
    }

    public var melFrames: Int {
        switch self {
        case .fiveSeconds: return 501
        case .tenSeconds: return 1001
        case .fifteenSeconds: return 1501
        case .twentySeconds: return 2001
        }
    }

    public var encodedFrames: Int {
        switch self {
        case .fiveSeconds: return 126
        case .tenSeconds: return 251
        case .fifteenSeconds: return 376
        case .twentySeconds: return 501
        }
    }

    public var seconds: Double {
        switch self {
        case .fiveSeconds: return 5.0
        case .tenSeconds: return 10.0
        case .fifteenSeconds: return 15.0
        case .twentySeconds: return 20.0
        }
    }

    /// Select appropriate duration for given sample count
    public static func select(forSamples count: Int) -> ConformerInputDuration {
        // Select the smallest duration that fits, or twentySeconds for longer
        for duration in allCases {
            if count <= duration.samples {
                return duration
            }
        }
        return .twentySeconds
    }
}

/// NeMo Conformer CTC ASR
public final class NeMoConformerASR: @unchecked Sendable {

    /// Sample rate expected by the model
    public static let sampleRate: Int = 16000

    /// Feature extractor
    private let featureExtractor: NeMoFeatureExtractor

    /// Encoder CoreML model
    private let encoder: MLModel

    /// Decoder CoreML model
    private let decoder: MLModel

    /// BPE vocabulary
    private let vocabulary: [String]

    /// CTC blank token ID
    private let blankId: Int = 1024

    /// Initialize NeMo Conformer ASR
    /// - Parameters:
    ///   - encoderURL: URL to conformer_encoder.mlmodelc
    ///   - decoderURL: URL to conformer_decoder.mlmodelc
    ///   - vocabularyURL: URL to vocabulary.json
    ///   - computeUnits: CoreML compute units to use
    public init(
        encoderURL: URL,
        decoderURL: URL,
        vocabularyURL: URL,
        computeUnits: MLComputeUnits = .all
    ) throws {
        // Initialize feature extractor with NeMo ASR config
        self.featureExtractor = NeMoFeatureExtractor(config: .nemoASR)

        // Load CoreML models
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        do {
            self.encoder = try MLModel(contentsOf: encoderURL, configuration: config)
        } catch {
            throw NeMoConformerASRError.modelLoadFailed("Encoder: \(error.localizedDescription)")
        }

        do {
            self.decoder = try MLModel(contentsOf: decoderURL, configuration: config)
        } catch {
            throw NeMoConformerASRError.modelLoadFailed("Decoder: \(error.localizedDescription)")
        }

        // Load vocabulary
        do {
            let vocabData = try Data(contentsOf: vocabularyURL)
            self.vocabulary = try JSONDecoder().decode([String].self, from: vocabData)
        } catch {
            throw NeMoConformerASRError.vocabularyLoadFailed(error.localizedDescription)
        }
    }

    // MARK: - Public API

    /// Recognize speech from audio samples
    /// - Parameter samples: Audio samples (Float32, mono, 16kHz)
    /// - Returns: Recognized text
    public func recognize(samples: [Float]) throws -> String {
        guard !samples.isEmpty else {
            throw NeMoConformerASRError.invalidInput("Empty audio samples")
        }

        // Process in chunks if audio is longer than max duration
        let maxSamples = ConformerInputDuration.twentySeconds.samples

        if samples.count <= maxSamples {
            // Single chunk processing
            let logits = try processChunk(samples)
            return ctcGreedyDecode(logits: logits)
        } else {
            // Multi-chunk processing with overlap
            return try processLongAudio(samples)
        }
    }

    /// Encode audio samples to encoder output
    /// - Parameter samples: Audio samples (Float32, mono, 16kHz)
    /// - Returns: Encoder output as MLMultiArray (1, 176, encodedFrames)
    public func encode(samples: [Float]) throws -> MLMultiArray {
        guard !samples.isEmpty else {
            throw NeMoConformerASRError.invalidInput("Empty audio samples")
        }

        // Pad to supported duration
        let paddedSamples = padToSupportedDuration(samples)
        let duration = ConformerInputDuration.select(forSamples: paddedSamples.count)

        // Extract mel features
        let melArray = try featureExtractor.processToMLMultiArray(samples: paddedSamples)

        // Verify mel shape
        let melFrames = melArray.shape[2].intValue
        guard melFrames == duration.melFrames else {
            throw NeMoConformerASRError.invalidInput(
                "Unexpected mel frames: \(melFrames), expected \(duration.melFrames)"
            )
        }

        // Run encoder
        let lengthArray = try createLengthArray(value: melFrames)

        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "mel_spectrogram": MLFeatureValue(multiArray: melArray),
            "length": MLFeatureValue(multiArray: lengthArray)
        ])

        let encoderOutput = try encoder.prediction(from: encoderInput)

        guard let encodedValue = encoderOutput.featureValue(for: "encoded"),
              let encoded = encodedValue.multiArrayValue else {
            throw NeMoConformerASRError.inferenceFailed("Failed to get encoder output")
        }

        return encoded
    }

    // MARK: - Private Methods

    /// Process a single audio chunk and return logits
    private func processChunk(_ samples: [Float]) throws -> [[Float]] {
        // Pad to supported duration
        let paddedSamples = padToSupportedDuration(samples)
        let duration = ConformerInputDuration.select(forSamples: paddedSamples.count)

        // Extract mel features
        let melArray = try featureExtractor.processToMLMultiArray(samples: paddedSamples)

        // Verify mel shape
        let melFrames = melArray.shape[2].intValue
        guard melFrames == duration.melFrames else {
            throw NeMoConformerASRError.invalidInput(
                "Unexpected mel frames: \(melFrames), expected \(duration.melFrames)"
            )
        }

        // Run encoder
        let lengthArray = try createLengthArray(value: melFrames)

        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "mel_spectrogram": MLFeatureValue(multiArray: melArray),
            "length": MLFeatureValue(multiArray: lengthArray)
        ])

        let encoderOutput = try encoder.prediction(from: encoderInput)

        guard let encodedValue = encoderOutput.featureValue(for: "encoded"),
              let encoded = encodedValue.multiArrayValue else {
            throw NeMoConformerASRError.inferenceFailed("Failed to get encoder output")
        }

        // Run decoder
        let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_output": MLFeatureValue(multiArray: encoded)
        ])

        let decoderOutput = try decoder.prediction(from: decoderInput)

        guard let logitsValue = decoderOutput.featureValue(for: "logits"),
              let logits = logitsValue.multiArrayValue else {
            throw NeMoConformerASRError.inferenceFailed("Failed to get decoder output")
        }

        // Convert to 2D array [time, vocab]
        return extractLogits(from: logits)
    }

    /// Process long audio by chunking
    private func processLongAudio(_ samples: [Float]) throws -> String {
        let chunkDuration = ConformerInputDuration.twentySeconds
        let chunkSamples = chunkDuration.samples
        let hopSamples = chunkSamples / 2  // 50% overlap

        var allTexts: [String] = []
        var offset = 0

        while offset < samples.count {
            let endIndex = min(offset + chunkSamples, samples.count)
            var chunk = Array(samples[offset..<endIndex])

            // Pad last chunk if needed
            if chunk.count < chunkSamples {
                chunk.append(contentsOf: [Float](repeating: 0, count: chunkSamples - chunk.count))
            }

            let logits = try processChunk(chunk)
            let text = ctcGreedyDecode(logits: logits)

            if !text.isEmpty {
                allTexts.append(text)
            }

            offset += hopSamples

            // Break if we've processed all audio
            if endIndex >= samples.count {
                break
            }
        }

        // Simple concatenation (could be improved with overlap handling)
        return allTexts.joined(separator: " ")
    }

    /// Pad samples to a supported duration
    private func padToSupportedDuration(_ samples: [Float]) -> [Float] {
        let duration = ConformerInputDuration.select(forSamples: samples.count)

        if samples.count == duration.samples {
            return samples
        }

        // Pad with zeros
        var padded = samples
        padded.append(contentsOf: [Float](repeating: 0, count: duration.samples - samples.count))
        return padded
    }

    /// Create length MLMultiArray
    private func createLengthArray(value: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1], dataType: .int32)
        array[0] = NSNumber(value: value)
        return array
    }

    /// Extract logits from MLMultiArray to 2D array
    private func extractLogits(from mlArray: MLMultiArray) -> [[Float]] {
        // Shape: [1, time, vocab_size]
        let timeSteps = mlArray.shape[1].intValue
        let vocabSize = mlArray.shape[2].intValue

        var logits: [[Float]] = []
        logits.reserveCapacity(timeSteps)

        for t in 0..<timeSteps {
            var frame = [Float](repeating: 0, count: vocabSize)
            for v in 0..<vocabSize {
                // Use 3D subscript access
                frame[v] = mlArray[[0, t, v] as [NSNumber]].floatValue
            }
            logits.append(frame)
        }

        return logits
    }

    /// CTC greedy decode
    private func ctcGreedyDecode(logits: [[Float]]) -> String {
        var decodedIds: [Int] = []
        var prevId: Int? = nil

        for frame in logits {
            // Find argmax
            var maxIdx = 0
            var maxVal = frame[0]
            for (idx, val) in frame.enumerated() {
                if val > maxVal {
                    maxVal = val
                    maxIdx = idx
                }
            }

            // Skip blanks and repeated tokens
            if maxIdx != blankId && maxIdx != prevId {
                decodedIds.append(maxIdx)
            }
            prevId = maxIdx
        }

        // Convert to text
        let tokens = decodedIds.compactMap { id -> String? in
            guard id < vocabulary.count else { return nil }
            return vocabulary[id]
        }

        // Join and clean up BPE tokens
        return tokens.joined().replacingOccurrences(of: "▁", with: " ").trimmingCharacters(in: .whitespaces)
    }
}

// MARK: - Convenience Extensions

extension NeMoConformerASR {
    /// Process Double samples
    public func recognize(samples: [Double]) throws -> String {
        let floatSamples = samples.map { Float($0) }
        return try recognize(samples: floatSamples)
    }

    /// Encode Double samples
    public func encode(samples: [Double]) throws -> MLMultiArray {
        let floatSamples = samples.map { Float($0) }
        return try encode(samples: floatSamples)
    }
}
