import Foundation
import CoreML
import NeMoFeatureExtractor
import NeMoVAD

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

/// A single recognized speech segment with timing information
public struct ASRSegment: Sendable {
    /// Start time in seconds
    public let start: Double
    /// End time in seconds
    public let end: Double
    /// Recognized text for this segment
    public let text: String

    /// Duration in seconds
    public var duration: Double { end - start }
}

/// Result of speech recognition
public struct ASRResult: Sendable {
    /// Full recognized text (all segments joined)
    public let text: String
    /// Individual segments with timing
    public let segments: [ASRSegment]
    /// Total audio duration in seconds
    public let audioDuration: Double
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
        for duration in allCases {
            if count <= duration.samples {
                return duration
            }
        }
        return .twentySeconds
    }
}

/// NeMo Conformer CTC ASR with VAD-based segmentation
public final class NeMoConformerASR: @unchecked Sendable {

    /// Sample rate expected by the model
    public static let sampleRate: Int = 16000

    /// Maximum duration for single inference (seconds)
    public static let maxSegmentDuration: Double = 20.0

    /// Minimum segment duration to process (seconds)
    public static let minSegmentDuration: Double = 0.5

    /// Minimum silence duration to use as cut point (seconds)
    public static let minSilenceForCut: Double = 0.3

    // MARK: - Private Properties

    private let featureExtractor: NeMoFeatureExtractor
    private let encoder: MLModel
    private let decoder: MLModel
    private let vocabulary: [String]
    private let vad: NeMoVAD
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
        // Initialize feature extractor
        self.featureExtractor = NeMoFeatureExtractor(config: .nemoASR)

        // Initialize VAD
        self.vad = try NeMoVAD(config: .default, computeUnits: computeUnits)

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
    /// - Returns: ASRResult with text and segments
    public func recognize(samples: [Float]) throws -> ASRResult {
        guard !samples.isEmpty else {
            throw NeMoConformerASRError.invalidInput("Empty audio samples")
        }

        let audioDuration = Double(samples.count) / Double(Self.sampleRate)

        // For short audio, process directly
        if audioDuration <= Self.maxSegmentDuration {
            let text = try recognizeSegment(samples: samples)
            let segment = ASRSegment(start: 0, end: audioDuration, text: text)
            return ASRResult(
                text: text,
                segments: text.isEmpty ? [] : [segment],
                audioDuration: audioDuration
            )
        }

        // For long audio, use VAD-based segmentation
        return try recognizeLongAudio(samples: samples, audioDuration: audioDuration)
    }

    /// Encode audio samples to encoder output
    /// - Parameter samples: Audio samples (Float32, mono, 16kHz)
    /// - Returns: Encoder output as MLMultiArray (1, 176, encodedFrames)
    public func encode(samples: [Float]) throws -> MLMultiArray {
        guard !samples.isEmpty else {
            throw NeMoConformerASRError.invalidInput("Empty audio samples")
        }

        let paddedSamples = padToSupportedDuration(samples)
        let duration = ConformerInputDuration.select(forSamples: paddedSamples.count)

        let melArray = try featureExtractor.processToMLMultiArray(samples: paddedSamples)

        let melFrames = melArray.shape[2].intValue
        guard melFrames == duration.melFrames else {
            throw NeMoConformerASRError.invalidInput(
                "Unexpected mel frames: \(melFrames), expected \(duration.melFrames)"
            )
        }

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

    /// Recognize long audio using VAD-based segmentation
    private func recognizeLongAudio(samples: [Float], audioDuration: Double) throws -> ASRResult {
        // Step 1: Run VAD to find speech segments
        let vadResult = try vad.process(samples: samples)

        // If no speech detected
        guard !vadResult.segments.isEmpty else {
            return ASRResult(text: "", segments: [], audioDuration: audioDuration)
        }

        // Step 2: Process VAD segments - merge close ones, split long ones
        let processedSegments = processVADSegments(vadResult.segments, audioDuration: audioDuration)

        // Step 3: Recognize each segment
        var asrSegments: [ASRSegment] = []

        for segment in processedSegments {
            let startSample = Int(segment.start * Double(Self.sampleRate))
            let endSample = min(Int(segment.end * Double(Self.sampleRate)), samples.count)

            guard endSample > startSample else { continue }

            let segmentSamples = Array(samples[startSample..<endSample])
            let text = try recognizeSegment(samples: segmentSamples)

            if !text.isEmpty {
                asrSegments.append(ASRSegment(
                    start: segment.start,
                    end: segment.end,
                    text: text
                ))
            }
        }

        // Step 4: Join all text
        let fullText = asrSegments.map { $0.text }.joined(separator: " ")

        return ASRResult(
            text: fullText,
            segments: asrSegments,
            audioDuration: audioDuration
        )
    }

    /// Process VAD segments: merge close ones, split long ones
    private func processVADSegments(_ segments: [VADSegment], audioDuration: Double) -> [VADSegment] {
        guard !segments.isEmpty else { return [] }

        var result: [VADSegment] = []

        // First, merge segments that are very close together
        var current = segments[0]

        for i in 1..<segments.count {
            let next = segments[i]
            let gap = next.start - current.end

            // Merge if gap is small
            if gap < Self.minSilenceForCut {
                current = VADSegment(start: current.start, end: next.end)
            } else {
                result.append(current)
                current = next
            }
        }
        result.append(current)

        // Then, split segments that are too long
        var finalResult: [VADSegment] = []

        for segment in result {
            let duration = segment.end - segment.start

            if duration <= Self.maxSegmentDuration {
                // Segment is short enough
                if duration >= Self.minSegmentDuration {
                    finalResult.append(segment)
                }
            } else {
                // Need to split this segment
                let splitSegments = splitLongSegment(segment)
                finalResult.append(contentsOf: splitSegments)
            }
        }

        return finalResult
    }

    /// Split a segment that's longer than maxSegmentDuration
    private func splitLongSegment(_ segment: VADSegment) -> [VADSegment] {
        let duration = segment.end - segment.start
        let numChunks = Int(ceil(duration / Self.maxSegmentDuration))
        let chunkDuration = duration / Double(numChunks)

        var result: [VADSegment] = []

        for i in 0..<numChunks {
            let start = segment.start + Double(i) * chunkDuration
            let end = min(segment.start + Double(i + 1) * chunkDuration, segment.end)
            result.append(VADSegment(start: start, end: end))
        }

        return result
    }

    /// Recognize a single segment (must be <= 20 seconds)
    private func recognizeSegment(samples: [Float]) throws -> String {
        let paddedSamples = padToSupportedDuration(samples)
        let duration = ConformerInputDuration.select(forSamples: paddedSamples.count)

        // Extract mel features
        let melArray = try featureExtractor.processToMLMultiArray(samples: paddedSamples)

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

        // CTC decode
        let logitsArray = extractLogits(from: logits)
        return ctcGreedyDecode(logits: logitsArray)
    }

    /// Pad samples to a supported duration
    private func padToSupportedDuration(_ samples: [Float]) -> [Float] {
        let duration = ConformerInputDuration.select(forSamples: samples.count)

        if samples.count == duration.samples {
            return samples
        }

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
        let timeSteps = mlArray.shape[1].intValue
        let vocabSize = mlArray.shape[2].intValue

        var logits: [[Float]] = []
        logits.reserveCapacity(timeSteps)

        for t in 0..<timeSteps {
            var frame = [Float](repeating: 0, count: vocabSize)
            for v in 0..<vocabSize {
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
            var maxIdx = 0
            var maxVal = frame[0]
            for (idx, val) in frame.enumerated() {
                if val > maxVal {
                    maxVal = val
                    maxIdx = idx
                }
            }

            if maxIdx != blankId && maxIdx != prevId {
                decodedIds.append(maxIdx)
            }
            prevId = maxIdx
        }

        let tokens = decodedIds.compactMap { id -> String? in
            guard id < vocabulary.count else { return nil }
            return vocabulary[id]
        }

        return tokens.joined().replacingOccurrences(of: "▁", with: " ").trimmingCharacters(in: .whitespaces)
    }
}

// MARK: - Convenience Extensions

extension NeMoConformerASR {
    /// Process Double samples
    public func recognize(samples: [Double]) throws -> ASRResult {
        let floatSamples = samples.map { Float($0) }
        return try recognize(samples: floatSamples)
    }

    /// Encode Double samples
    public func encode(samples: [Double]) throws -> MLMultiArray {
        let floatSamples = samples.map { Float($0) }
        return try encode(samples: floatSamples)
    }
}
