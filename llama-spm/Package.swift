// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "llama",
    platforms: [.iOS(.v16)],
    products: [
        // Your app should depend on this library
        .library(name: "llama", targets: ["llama"])
    ],
    targets: [
        // Core llama runtime (text)
        .binaryTarget(
            name: "llama_bin",
            path: "Frameworks/llama.xcframework"
        ),

        // Multimodal runtime (vision)
        .binaryTarget(
            name: "mtmd_bin",
            path: "Frameworks/mtmd.xcframework"
        ),

        // GGML math backend (link-only, headers removed to avoid collisions)
        .binaryTarget(
            name: "ggml_bin",
            path: "Frameworks/ggml.xcframework"
        ),

        // C module that exposes llama.h + mtmd.h via umbrella header
        .target(
            name: "llama_c",
            dependencies: ["llama_bin", "mtmd_bin", "ggml_bin"],
            path: "Sources/llama_c",
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("../../Frameworks/llama.xcframework/ios-arm64/Headers"),
                .headerSearchPath("../../Frameworks/llama.xcframework/ios-arm64_x86_64-simulator/Headers"),
                .headerSearchPath("../../Frameworks/mtmd.xcframework/ios-arm64/Headers"),
                .headerSearchPath("../../Frameworks/mtmd.xcframework/ios-arm64_x86_64-simulator/Headers"),
                .headerSearchPath("../../Headers/ggml")
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit")
            ]
        ),

        // Swift wrapper for app import
        .target(
            name: "llama",
            dependencies: ["llama_c", "llama_bin", "mtmd_bin", "ggml_bin"],
            path: "Sources/llama",
            sources: ["llama.swift"]
        )
    ]
)
