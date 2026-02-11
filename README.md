# NeMoConformerASR-iOS

Swift library for speech recognition using NVIDIA NeMo Conformer CTC model on iOS/macOS with CoreML.

## Features

- NVIDIA NeMo Conformer CTC Small model (13M parameters)
- **VAD-based smart segmentation** for long audio (powered by [NeMoVAD-iOS](https://github.com/Otosaku/NeMoVAD-iOS))
- Returns both full text and timestamped segments
- Automatic audio padding for any duration
- Support for 5, 10, 15, and 20 second audio segments
- Pure Swift implementation with CoreML backend

## Requirements

- iOS 16.0+ / macOS 13.0+
- Xcode 15.0+
- Swift 5.9+

## Installation

### Swift Package Manager

Add the following to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/Otosaku/NeMoConformerASR-iOS.git", from: "1.1.0")
]
```

> **Note:** Version 1.1.0+ includes VAD-based segmentation with timestamped results. For the previous API returning plain text, use version 1.0.0.

Or in Xcode: File → Add Package Dependencies → Enter repository URL.

### Download Models

Download the CoreML models from Google Drive:

**[Download Models (30 MB)](https://drive.google.com/file/d/1iG1dln3Sp9k_TjXJPVs4LAWbX-U0jeI3/view?usp=sharing)**

The archive contains:
- `conformer_encoder.mlmodelc` - Conformer encoder (30 MB)
- `conformer_decoder.mlmodelc` - CTC decoder (0.4 MB)
- `vocabulary.json` - BPE vocabulary (1024 tokens)

## Usage

### Basic Recognition

```swift
import NeMoConformerASR

// Initialize with model paths
let asr = try NeMoConformerASR(
    encoderURL: Bundle.main.url(forResource: "conformer_encoder", withExtension: "mlmodelc")!,
    decoderURL: Bundle.main.url(forResource: "conformer_decoder", withExtension: "mlmodelc")!,
    vocabularyURL: Bundle.main.url(forResource: "vocabulary", withExtension: "json")!,
    computeUnits: .all  // .cpuAndGPU, .cpuOnly, .cpuAndNeuralEngine
)

// Recognize speech (samples must be 16kHz mono Float32)
let result = try asr.recognize(samples: audioSamples)

// Full recognized text
print(result.text)

// Individual segments with timestamps
for segment in result.segments {
    print("[\(segment.start)s - \(segment.end)s]: \(segment.text)")
}

// Audio duration
print("Duration: \(result.audioDuration)s")
```

### ASRResult Structure

```swift
public struct ASRResult {
    let text: String           // Full recognized text
    let segments: [ASRSegment] // Timestamped segments
    let audioDuration: Double  // Total audio duration in seconds
}

public struct ASRSegment {
    let start: Double  // Start time in seconds
    let end: Double    // End time in seconds
    let text: String   // Recognized text for this segment
}
```

### Get Encoder Output

```swift
// Get encoder embeddings for downstream tasks
let encoded = try asr.encode(samples: audioSamples)
// Returns MLMultiArray with shape [1, 176, encodedFrames]
```

### Supported Input Durations

The model supports the following input sizes (audio is automatically padded):

| Duration | Samples | Mel Frames | Encoded Frames |
|----------|---------|------------|----------------|
| 5 sec    | 80,000  | 501        | 126            |
| 10 sec   | 160,000 | 1,001      | 251            |
| 15 sec   | 240,000 | 1,501      | 376            |
| 20 sec   | 320,000 | 2,001      | 501            |

### Long Audio Processing

For audio longer than 20 seconds, the library uses VAD (Voice Activity Detection) for intelligent segmentation:

1. **VAD Analysis**: Detects speech vs silence regions
2. **Smart Merging**: Merges speech segments with gaps < 0.3s
3. **Splitting**: Splits segments longer than 20s into equal parts
4. **Filtering**: Ignores segments shorter than 0.5s
5. **Recognition**: Processes each segment independently

This approach provides accurate timestamps and avoids cutting words in the middle.

## Example Project

The repository includes a complete example app with audio recording and file import.

### Running the Example

1. Open `ConformerExample/ConformerExample.xcodeproj` in Xcode

2. Add NeMoConformerASR as a local package:
   - File → Add Package Dependencies
   - Click "Add Local..."
   - Select the `NeMoConformerASR-iOS` folder

3. Download and add models to the project:
   - Download models from the link above
   - Unzip the archive
   - Drag `conformer_encoder.mlmodelc`, `conformer_decoder.mlmodelc`, and `vocabulary.json` into `ConformerExample/Resources` folder in Xcode
   - Make sure "Copy items if needed" is checked
   - Verify files are added to "Copy Bundle Resources" in Build Phases

4. Build and run on device or simulator

### Example Features

- **Record Audio**: Tap to record from microphone, automatically converts to 16kHz mono
- **Import Audio**: Import any audio file (mp3, wav, m4a, etc.), automatically converts format
- **Results**: Shows recognized text, audio duration, and processing time
- **Segments View**: Displays individual speech segments with timestamps for long audio

## Model Information

- **Model**: nvidia/stt_en_conformer_ctc_small
- **Parameters**: 13.15M
- **Architecture**: Conformer encoder (16 layers) + CTC decoder
- **Hidden dim**: 176
- **Attention heads**: 4
- **Vocabulary**: 1024 BPE tokens + 1 blank

## Audio Requirements

- Sample rate: 16,000 Hz
- Channels: Mono
- Format: Float32

The example app handles conversion from any audio format automatically.

## Dependencies

- [NeMoFeatureExtractor-iOS](https://github.com/Otosaku/NeMoFeatureExtractor-iOS) - Mel spectrogram extraction
- [NeMoVAD-iOS](https://github.com/Otosaku/NeMoVAD-iOS) - Voice Activity Detection for smart segmentation

## License

MIT License

## Acknowledgments

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) - Original model and training
