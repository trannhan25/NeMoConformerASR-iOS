//
//  ContentView.swift
//  ConformerExample
//

import SwiftUI
import NeMoConformerASR
import AVFoundation
import UniformTypeIdentifiers
internal import Combine
internal import CoreML

struct ContentView: View {
    @StateObject private var viewModel = ASRViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                // Status
                statusView

                // Recording controls
                recordingControls

                // Import button
                importButton

                // Result
                if !viewModel.recognizedText.isEmpty {
                    resultView
                }

                Spacer()
            }
            .padding()
            .navigationTitle("Conformer ASR")
            .fileImporter(
                isPresented: $viewModel.showFilePicker,
                allowedContentTypes: [.audio],
                allowsMultipleSelection: false
            ) { result in
                viewModel.handleFileImport(result)
            }
            .alert("Error", isPresented: $viewModel.showError) {
                Button("OK") {}
            } message: {
                Text(viewModel.errorMessage)
            }
        }
    }

    private var statusView: some View {
        VStack(spacing: 8) {
            if viewModel.isLoading {
                ProgressView("Loading models...")
            } else if viewModel.isProcessing {
                ProgressView("Recognizing...")
            } else if viewModel.isRecording {
                HStack {
                    Circle()
                        .fill(.red)
                        .frame(width: 12, height: 12)
                    Text("Recording: \(viewModel.recordingDuration, specifier: "%.1f")s")
                        .monospacedDigit()
                }
                .font(.headline)
            } else if viewModel.isReady {
                Label("Ready", systemImage: "checkmark.circle.fill")
                    .foregroundColor(.green)
            }
        }
        .frame(height: 44)
    }

    private var recordingControls: some View {
        VStack(spacing: 16) {
            if viewModel.isRecording {
                // Stop button
                Button(action: viewModel.stopRecording) {
                    Label("Stop Recording", systemImage: "stop.circle.fill")
                        .font(.title2)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(.red)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }
            } else {
                // Record button
                Button(action: viewModel.startRecording) {
                    Label("Record Audio", systemImage: "mic.circle.fill")
                        .font(.title2)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(viewModel.isReady ? .blue : .gray)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }
                .disabled(!viewModel.isReady)
            }
        }
    }

    private var importButton: some View {
        Button(action: { viewModel.showFilePicker = true }) {
            Label("Import Audio File", systemImage: "doc.badge.plus")
                .font(.title3)
                .frame(maxWidth: .infinity)
                .padding()
                .background(viewModel.isReady ? Color.blue.opacity(0.1) : .gray.opacity(0.1))
                .foregroundColor(viewModel.isReady ? .blue : .gray)
                .cornerRadius(12)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(viewModel.isReady ? .blue : .gray, lineWidth: 1)
                )
        }
        .disabled(!viewModel.isReady)
    }

    private var resultView: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Result")
                    .font(.headline)
                Spacer()
                Button(action: {
                    UIPasteboard.general.string = viewModel.recognizedText
                }) {
                    Image(systemName: "doc.on.doc")
                }
            }

            Text(viewModel.recognizedText)
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.gray.opacity(0.1))
                .cornerRadius(8)

            if let duration = viewModel.audioDuration,
               let processingTime = viewModel.processingTime {
                HStack {
                    Text("Audio: \(duration, specifier: "%.1f")s")
                    Spacer()
                    Text("Processing: \(processingTime, specifier: "%.2f")s")
                }
                .font(.caption)
                .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color.green.opacity(0.05))
        .cornerRadius(12)
    }
}

// MARK: - ViewModel

@MainActor
class ASRViewModel: ObservableObject {
    @Published var isLoading = true
    @Published var isReady = false
    @Published var isRecording = false
    @Published var isProcessing = false
    @Published var recordingDuration: Double = 0
    @Published var recognizedText = ""
    @Published var showFilePicker = false
    @Published var showError = false
    @Published var errorMessage = ""
    @Published var audioDuration: Double?
    @Published var processingTime: Double?

    private var asr: NeMoConformerASR?
    private var audioRecorder: AudioRecorder?
    private var recordingTimer: Timer?

    init() {
        Task {
            await loadModels()
        }
    }

    private func loadModels() async {
        do {
            guard let encoderURL = Bundle.main.url(forResource: "conformer_encoder", withExtension: "mlmodelc"),
                  let decoderURL = Bundle.main.url(forResource: "conformer_decoder", withExtension: "mlmodelc"),
                  let vocabURL = Bundle.main.url(forResource: "vocabulary", withExtension: "json") else {
                throw NSError(domain: "ASR", code: 1, userInfo: [NSLocalizedDescriptionKey: "Models not found in bundle"])
            }

            asr = try NeMoConformerASR(
                encoderURL: encoderURL,
                decoderURL: decoderURL,
                vocabularyURL: vocabURL,
                computeUnits: .all
            )

            audioRecorder = AudioRecorder()

            isLoading = false
            isReady = true
        } catch {
            showError(error.localizedDescription)
            isLoading = false
        }
    }

    func startRecording() {
        guard let recorder = audioRecorder else { return }

        do {
            try recorder.startRecording()
            isRecording = true
            recordingDuration = 0
            recognizedText = ""

            recordingTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    self?.recordingDuration += 0.1
                }
            }
        } catch {
            showError(error.localizedDescription)
        }
    }

    func stopRecording() {
        recordingTimer?.invalidate()
        recordingTimer = nil

        guard let recorder = audioRecorder else { return }

        do {
            let samples = try recorder.stopRecording()
            isRecording = false

            if samples.isEmpty {
                showError("No audio recorded")
                return
            }

            recognize(samples: samples)
        } catch {
            isRecording = false
            showError(error.localizedDescription)
        }
    }

    func handleFileImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            loadAndRecognize(url: url)
        case .failure(let error):
            showError(error.localizedDescription)
        }
    }

    private func loadAndRecognize(url: URL) {
        isProcessing = true
        recognizedText = ""

        Task {
            do {
                let samples = try await loadAudioFile(url: url)
                audioDuration = Double(samples.count) / 16000.0

                recognize(samples: samples)
            } catch {
                isProcessing = false
                showError(error.localizedDescription)
            }
        }
    }

    private func loadAudioFile(url: URL) async throws -> [Float] {
        // Start accessing security-scoped resource
        guard url.startAccessingSecurityScopedResource() else {
            throw NSError(domain: "ASR", code: 2, userInfo: [NSLocalizedDescriptionKey: "Cannot access file"])
        }
        defer { url.stopAccessingSecurityScopedResource() }

        let file = try AVAudioFile(forReading: url)
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!

        guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw NSError(domain: "ASR", code: 3, userInfo: [NSLocalizedDescriptionKey: "Cannot create buffer"])
        }

        try file.read(into: buffer)

        // Convert to 16kHz mono
        guard let converter = AVAudioConverter(from: file.processingFormat, to: format) else {
            throw NSError(domain: "ASR", code: 4, userInfo: [NSLocalizedDescriptionKey: "Cannot create converter"])
        }

        let outputFrameCount = AVAudioFrameCount(Double(buffer.frameLength) * 16000.0 / file.processingFormat.sampleRate)
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: outputFrameCount) else {
            throw NSError(domain: "ASR", code: 5, userInfo: [NSLocalizedDescriptionKey: "Cannot create output buffer"])
        }

        var error: NSError?
        converter.convert(to: outputBuffer, error: &error) { inNumPackets, outStatus in
            outStatus.pointee = .haveData
            return buffer
        }

        if let error = error {
            throw error
        }

        guard let floatData = outputBuffer.floatChannelData?[0] else {
            throw NSError(domain: "ASR", code: 6, userInfo: [NSLocalizedDescriptionKey: "Cannot get audio data"])
        }

        return Array(UnsafeBufferPointer(start: floatData, count: Int(outputBuffer.frameLength)))
    }

    private func recognize(samples: [Float]) {
        guard let asr = asr else { return }

        isProcessing = true
        audioDuration = Double(samples.count) / 16000.0

        Task {
            do {
                let startTime = CFAbsoluteTimeGetCurrent()
                let text = try asr.recognize(samples: samples)
                let endTime = CFAbsoluteTimeGetCurrent()

                processingTime = endTime - startTime
                recognizedText = text.isEmpty ? "(no speech detected)" : text
                isProcessing = false
            } catch {
                isProcessing = false
                showError(error.localizedDescription)
            }
        }
    }

    private func showError(_ message: String) {
        errorMessage = message
        showError = true
    }
}

// MARK: - Audio Recorder

class AudioRecorder {
    private var audioEngine: AVAudioEngine?
    private var samples: [Float] = []
    private let sampleRate: Double = 16000

    func startRecording() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement)
        try session.setActive(true)

        audioEngine = AVAudioEngine()
        guard let audioEngine = audioEngine else {
            throw NSError(domain: "Recorder", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot create audio engine"])
        }

        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        // Target format: 16kHz mono
        let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false)!

        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            throw NSError(domain: "Recorder", code: 2, userInfo: [NSLocalizedDescriptionKey: "Cannot create converter"])
        }

        samples = []

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, _ in
            self?.processBuffer(buffer, converter: converter, targetFormat: targetFormat)
        }

        audioEngine.prepare()
        try audioEngine.start()
    }

    private func processBuffer(_ buffer: AVAudioPCMBuffer, converter: AVAudioConverter, targetFormat: AVAudioFormat) {
        let frameCount = AVAudioFrameCount(Double(buffer.frameLength) * sampleRate / buffer.format.sampleRate)
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: frameCount) else { return }

        var error: NSError?
        converter.convert(to: outputBuffer, error: &error) { _, outStatus in
            outStatus.pointee = .haveData
            return buffer
        }

        guard error == nil,
              let floatData = outputBuffer.floatChannelData?[0] else { return }

        let newSamples = Array(UnsafeBufferPointer(start: floatData, count: Int(outputBuffer.frameLength)))
        samples.append(contentsOf: newSamples)
    }

    func stopRecording() throws -> [Float] {
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine = nil

        try AVAudioSession.sharedInstance().setActive(false)

        return samples
    }
}

#Preview {
    ContentView()
}
