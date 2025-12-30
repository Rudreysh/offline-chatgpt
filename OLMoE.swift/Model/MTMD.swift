import Foundation
import Darwin
import llama_c

// Fallback opaque types for Swift-facing code paths.
// These mirror the C opaque handles from mtmd.h.
typealias mtmd_context = OpaquePointer
typealias mtmd_bitmap = OpaquePointer
typealias mtmd_input_chunk = OpaquePointer
typealias mtmd_input_chunks = OpaquePointer
typealias mtmd_image_tokens = OpaquePointer

enum MTMDRuntime {
    static var isAvailable: Bool {
        let handle = dlopen(nil, RTLD_NOW)
        defer { if handle != nil { dlclose(handle) } }
        return dlsym(handle, "mtmd_init_from_file") != nil
    }
}

let MTMD_INPUT_CHUNK_TYPE_TEXT: Int32 = 0
let MTMD_INPUT_CHUNK_TYPE_IMAGE: Int32 = 1
let MTMD_INPUT_CHUNK_TYPE_AUDIO: Int32 = 2
