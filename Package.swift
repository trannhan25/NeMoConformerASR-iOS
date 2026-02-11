// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "NeMoConformerASR",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "NeMoConformerASR",
            targets: ["NeMoConformerASR"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/Otosaku/NeMoFeatureExtractor-iOS.git", from: "1.0.5"),
        .package(url: "https://github.com/Otosaku/NeMoVAD-iOS.git", from: "1.0.0")
    ],
    targets: [
        .target(
            name: "NeMoConformerASR",
            dependencies: [
                .product(name: "NeMoFeatureExtractor", package: "NeMoFeatureExtractor-iOS"),
                .product(name: "NeMoVAD", package: "NeMoVAD-iOS")
            ]
        ),
        .testTarget(
            name: "NeMoConformerASRTests",
            dependencies: ["NeMoConformerASR"],
            resources: [
                .process("Resources")
            ]
        ),
    ]
)
