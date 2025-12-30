# Implemented Features (OLMoE.swift)

This document is a living reference for re‑implementing the current feature set.

## Model Catalog + Persistence
- Central model registry: `OLMoE.swift/Model/ModelCatalog.swift`.
- Built‑in models:
  - OLMoE (Default) text model (from `AppConstants.Model`).
  - Gemma 2B (Instruct) text model.
  - SmolVLM2 2.2B (Vision) model + mmproj.
  - Qwen2‑VL 2B (Vision, Q8_0) model + mmproj.
- ModelStore persistence:
  - Stored in `Documents/models.json`.
  - Selected model persisted in UserDefaults (`selectedModelID`).
  - Custom models supported via `ModelStore.addCustomModel(_:)` and persisted.
  - Vision models are limited to the built‑in allow‑list (custom vision models are filtered out unless allowed).

## Download Manager (GGUF + mmproj)
- Download handled by `OLMoE.swift/Model/ModelDownloader.swift`.
- Vision models download **two artifacts**:
  - main GGUF
  - mmproj GGUF
- Artifacts are stored under the model folder:
  - `Documents/Models/<modelId>/<filename>`
- Features:
  - progress tracking per model
  - cancel download
  - delete local files
  - minimal size check (rejects files < 1MB)
- Errors are surfaced via `ModelDownloadState.error`.

## Chat UI + Navigation
- Chat view shows selected model name in the title.
- Model selection is surfaced from the model list view; selected model persisted.
- Back button is present on Chat to return to model selection.
- Share / action buttons in chat header are retained.

## Attachments
- Image + document attachments supported in chat composer.
- Attachments shown as preview cards before sending.
- If attachments are added on a non‑vision model, app prompts user to select a vision model (or OCR where applicable).

## Vision Model Support
- Vision models are marked with `supportsVision` and `kind = .vision`.
- Each vision model is paired with an mmproj file (downloaded automatically).
- Runtime path uses mtmd bridge (C API) to:
  - initialize mtmd with mmproj
  - create bitmap from RGB
  - tokenize + eval chunks
  - inject tokens into llama inference

## OCR Support
- `supportsOCR` flag is enabled for models.
- OCR flow can be layered on attachments when enabled (local extraction before prompt). 

## Error Handling + UX
- Download errors shown in model list cards.
- Attachment errors shown as alert popups.
- Copy‑error affordance is present (icon based) for error dialogs.

## Files to Re‑Check When Re‑Implementing
- Model catalog: `OLMoE.swift/Model/ModelCatalog.swift`
- Model persistence: `OLMoE.swift/Model/ModelStore.swift`
- Download logic: `OLMoE.swift/Model/ModelDownloader.swift`
- Chat runtime: `OLMoE.swift/Model/LLM.swift`
- Chat UI: `OLMoE.swift/Views/Chat/ChatView.swift`, `OLMoE.swift/Views/ContentView.swift`

## Vision Model URLs (Current)
- SmolVLM2 2.2B (Vision)
  - Model: https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/resolve/main/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf?download=true
  - mmproj: https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf?download=true
- Qwen2‑VL 2B (Vision, Q8_0)
  - Model: https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q8_0.gguf?download=true
  - mmproj: https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-f16.gguf?download=true
- Gemma 2B (Instruct)
  - Model: https://huggingface.co/ggml-org/gemma-2b-it-GGUF/resolve/main/gemma-2b-it-Q4_K_M.gguf?download=true

## Local Package: llama‑spm (mtmd‑enabled)
- Location: `OLMoE.swift/llama-spm/`
- Binary: `Frameworks/llama.xcframework`
- mtmd bridge target: `Sources/llama_mtmd/`
- Header search paths and vendor mtmd sources are required for vision.
- Audio sources are excluded; image path only.

## Known Runtime Requirements
- Vision requires:
  - mtmd symbols linked in app binary
  - mmproj present locally for the selected model
- Simulator: Metal can crash; set `GGML_METAL_DISABLE=1` for simulator runs.

