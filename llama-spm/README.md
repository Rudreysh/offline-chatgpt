# llama-spm

Local Swift Package wrapper for `llama.xcframework`.

## Structure

```
llama-spm/
  Package.swift
  Frameworks/llama.xcframework
  Sources/llama/llama.swift
  Sources/llama_mtmd/...
```

The app imports the module `llama` and does not need to know about `llama_bin` or `llama_mtmd`.

## Check MTMD availability

```
./scripts/check_mtmd.sh
```

If the output says **mtmd symbols NOT found**, the framework is text-only.

## Rebuild mtmd-enabled xcframework (manual)

From the `llama.cpp` repo root:

```
./build-xcframework.sh
```

This produces:

```
llama.cpp/build-apple/llama.xcframework
```

Copy it into:

```
OLMoE.swift/llama-spm/Frameworks/llama.xcframework
```

Then clean and rebuild the iOS app.

You can also run:

```
./scripts/build_llama_mtmd_xcframework.sh
```

from `OLMoE.swift/` to build and replace in one step.

## Adding vision models

For each vision model entry in the app model catalog:

- Provide both `model.gguf` and `mmproj-*.gguf` URLs.
- Download both files into the same model folder.
- Vision inference will only work if the xcframework contains mtmd symbols.
