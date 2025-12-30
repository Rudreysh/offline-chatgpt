import Foundation

struct ModelSpec: Identifiable {
    let id: String
    let displayName: String
    let filename: String
    let downloadURL: URL
    let template: Template
    let isMultimodal: Bool
    let projectorFilename: String?
    let projectorURL: URL?

    var localDirectory: URL {
        URL.modelsDirectory.appendingPathComponent(id)
    }

    var localModelURL: URL {
        localDirectory.appendingPathComponent(filename)
    }

    var localProjectorURL: URL? {
        guard let projectorFilename else { return nil }
        return localDirectory.appendingPathComponent(projectorFilename)
    }

    var isModelDownloaded: Bool {
        localModelURL.isValidGGUF
    }

    var isProjectorDownloaded: Bool {
        guard let projectorURL = localProjectorURL else { return true }
        return projectorURL.isValidGGUF
    }
}

private extension URL {
    var isValidGGUF: Bool {
        guard exists else { return false }
        guard let size = (try? resourceValues(forKeys: [.fileSizeKey]).fileSize), size > 1_000_000 else {
            return false
        }
        guard let handle = try? FileHandle(forReadingFrom: self) else { return false }
        defer { try? handle.close() }
        let magic = handle.readData(ofLength: 4)
        return String(data: magic, encoding: .utf8) == "GGUF"
    }
}

enum ModelCatalog {
    static let models: [ModelSpec] = [
        ModelSpec(
            id: "olmoe-latest",
            displayName: "OLMoE (Text)",
            filename: "OLMoE-latest.gguf",
            downloadURL: URL(string: "https://dolma-artifacts.org/app/OLMoE-latest.gguf")!,
            template: .OLMoE(),
            isMultimodal: false,
            projectorFilename: nil,
            projectorURL: nil
        ),
        ModelSpec(
            id: "tinyllama-1-1b-chat",
            displayName: "TinyLlama 1.1B (Chat)",
            filename: "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
            downloadURL: URL(string: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true")!,
            template: .chatML(),
            isMultimodal: false,
            projectorFilename: nil,
            projectorURL: nil
        ),
        ModelSpec(
            id: "phi-3-mini-4k-instruct-q4",
            displayName: "Phi-3 Mini 4K (Instruct, Q4)",
            filename: "Phi-3-mini-4k-instruct-q4.gguf",
            downloadURL: URL(string: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf?download=true")!,
            template: .chatML(),
            isMultimodal: false,
            projectorFilename: nil,
            projectorURL: nil
        ),
        ModelSpec(
            id: "openelm-1-1b-instruct-q8",
            displayName: "OpenELM 1.1B Instruct",
            filename: "OpenELM-1_1B-Instruct-Q8_0.gguf",
            downloadURL: URL(string: "https://huggingface.co/mradermacher/OpenELM-1_1B-Instruct-GGUF/resolve/main/OpenELM-1_1B-Instruct.Q8_0.gguf?download=true")!,
            template: .chatML(),
            isMultimodal: false,
            projectorFilename: nil,
            projectorURL: nil
        ),
        ModelSpec(
            id: "openelm-270m-instruct-f16",
            displayName: "OpenELM 270M Instruct (F16)",
            filename: "OpenELM-270M-Instruct.f16.gguf",
            downloadURL: URL(string: "https://huggingface.co/mradermacher/OpenELM-270M-Instruct-GGUF/resolve/main/OpenELM-270M-Instruct.f16.gguf?download=true")!,
            template: .chatML(),
            isMultimodal: false,
            projectorFilename: nil,
            projectorURL: nil
        ),
        ModelSpec(
            id: "lfm2-350m-q8",
            displayName: "LFM2 350M (Q8_0)",
            filename: "LFM2-350M-Q8_0.gguf",
            downloadURL: URL(string: "https://huggingface.co/LiquidAI/LFM2-350M-GGUF/resolve/main/LFM2-350M-Q8_0.gguf?download=true")!,
            template: .chatML(),
            isMultimodal: false,
            projectorFilename: nil,
            projectorURL: nil
        ),
        ModelSpec(
            id: "smolvlm2-2b",
            displayName: "SmolVLM2 2.2B (Vision)",
            filename: "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
            downloadURL: URL(string: "https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/resolve/main/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf?download=true")!,
            template: .chatML(),
            isMultimodal: true,
            projectorFilename: "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf",
            projectorURL: URL(string: "https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf?download=true")
        ),
        ModelSpec(
            id: "qwen2vl-2b-q8",
            displayName: "Qwen2-VL 2B (Vision, Q8_0)",
            filename: "Qwen2-VL-2B-Instruct-Q8_0.gguf",
            downloadURL: URL(string: "https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q8_0.gguf?download=true")!,
            template: .chatML(),
            isMultimodal: true,
            projectorFilename: "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf",
            projectorURL: URL(string: "https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf?download=true")
        ),
        ModelSpec(
            id: "qwen3vl-2b-q8",
            displayName: "Qwen3-VL 2B (Vision, Q8_0)",
            filename: "Qwen3VL-2B-Instruct-Q8_0.gguf",
            downloadURL: URL(string: "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-Q8_0.gguf?download=true")!,
            template: .chatML(),
            isMultimodal: true,
            projectorFilename: "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
            projectorURL: URL(string: "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf?download=true")
        ),
        ModelSpec(
            id: "ministral-3-3b-instruct-q2",
            displayName: "Ministral 3 3B (Vision, Q2_K_L)",
            filename: "Ministral-3-3B-Instruct-2512-Q2_K_L.gguf",
            downloadURL: URL(string: "https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q2_K_L.gguf?download=true")!,
            template: .chatML(),
            isMultimodal: true,
            projectorFilename: "mmproj-BF16.gguf",
            projectorURL: URL(string: "https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/mmproj-BF16.gguf?download=true")
        ),
        ModelSpec(
            id: "llava-phi-3-mini-q8",
            displayName: "LLaVA-Phi-3 Mini (Vision, Q8_0)",
            filename: "llava-phi-3-mini-instruct-q8_0.gguf",
            downloadURL: URL(string: "https://huggingface.co/bartowski/LLaVA-Phi-3-mini-GGUF/resolve/main/llava-phi-3-mini-instruct-q8_0.gguf?download=true")!,
            template: .chatML(),
            isMultimodal: true,
            projectorFilename: "mmproj-llava-phi-3-mini-f16.gguf",
            projectorURL: URL(string: "https://huggingface.co/bartowski/LLaVA-Phi-3-mini-GGUF/resolve/main/mmproj-llava-phi-3-mini-f16.gguf?download=true")
        )
    ]
}
