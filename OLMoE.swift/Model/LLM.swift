import Foundation
import UIKit
import llama
import llama_c


public typealias Token = llama_token
public typealias Model = OpaquePointer

public struct Chat: Identifiable, Equatable {
    public var id: UUID? // Optional unique identifier
    public var role: Role
    public var content: String

    public init(id: UUID? = UUID(), role: Role, content: String) {
        self.id = id
        self.role = role
        self.content = content
    }
}

/// An actor that manages access to LLM inference operations to ensure thread safety
@globalActor public actor InferenceActor {
    static public let shared = InferenceActor()
}

/// Base class for Large Language Model inference
/// Provides functionality for text generation, chat history management, and model state control
open class LLM: ObservableObject {
    /// The underlying LLaMA model pointer
    public var model: Model

    /// Array of chat messages representing the conversation history
    public var history: [Chat]

    /// Closure to preprocess input before sending to the model
    /// - Parameters:
    ///   - input: The raw input string
    ///   - history: Current chat history
    ///   - llmInstance: Reference to the LLM instance
    /// - Returns: Processed input string ready for the model
    public var preprocess: (_ input: String, _ history: [Chat], _ llmInstance: LLM) -> String = { input, _, _ in return input }

    /// Closure called when generation is complete with the final output
    /// - Parameter output: The complete generated response
    public var postprocess: (_ output: String) -> Void = { print($0) }

    /// Closure called during generation with incremental output
    /// - Parameter outputDelta: New text fragment (nil when generation ends)
    public var update: (_ outputDelta: String?) -> Void = { _ in }

    /// Template controlling model input/output formatting
    /// Setting this updates preprocess and stop sequence configuration
    public var template: Template? = nil {
        didSet {
            guard let template else {
                preprocess = { input, _, _ in return input }
                stopSequence = nil
                stopSequenceLength = 0
                return
            }
            preprocess = template.preprocess
            if let stopSequence = template.stopSequence?.utf8CString {
                self.stopSequence = stopSequence
                stopSequenceLength = stopSequence.count - 1
            } else {
                stopSequence = nil
                stopSequenceLength = 0
            }
        }
    }

    /// Top-K sampling parameter - limits vocabulary to K most likely tokens
    public var topK: Int32

    /// Top-P sampling parameter - limits vocabulary to tokens comprising top P probability mass
    public var topP: Float

    /// Temperature parameter controlling randomness of sampling (higher = more random)
    public var temp: Float

    /// Path to the model file
    public var path: [CChar]

    /// Flag to enable test response mode
    public var loopBackTestResponse: Bool = false

    /// Cached model state for continuation of conversations
    public var savedState: Data?

    /// Metrics for tracking inference performance and token counts
    public var metrics = InferenceMetrics()

    /// Current generated output text
    @Published public private(set) var output = ""
    @MainActor public func setOutput(to newOutput: consuming String) {
        output = newOutput.trimmingCharacters(in: .whitespaces)
    }

    private var batch: llama_batch!
    private var context: Context!
    private var decoded = ""
    private var inferenceTask: Task<Void, Never>?
    private var input: String = ""
    private var isAvailable = true
    private let newlineToken: Token
    private let maxTokenCount: Int
    private var multibyteCharacter: [CUnsignedChar] = []
    private var params: llama_context_params
    private var sampler: UnsafeMutablePointer<llama_sampler>?
    private var stopSequence: ContiguousArray<CChar>?
    private var stopSequenceLength: Int
    private let totalTokenCount: Int
    private var updateProgress: (Double) -> Void = { _ in }
    private var nPast: Int32 = 0 // Track number of tokens processed
    private var inputTokenCount: Int32 = 0
    private var mtmdContext: mtmd_context?
    private var mtmdProjectorPath: String?

    public init(
        from path: String,
        stopSequence: String? = nil,
        history: [Chat] = [],
        seed: UInt32 = .random(in: .min ... .max),
        topK: Int32 = 40,
        topP: Float = 0.95,
        temp: Float = 0.8,
        maxTokenCount: Int32 = 2048
    ) {
        self.path = path.cString(using: .utf8)!
        var modelParams = llama_model_default_params()
        #if targetEnvironment(simulator)
            modelParams.n_gpu_layers = 0
        #endif
        let model = llama_model_load_from_file(self.path, modelParams)!
        self.params = llama_context_default_params()
        let processorCount = Int32(ProcessInfo().processorCount)
        self.maxTokenCount = Int(min(maxTokenCount, llama_model_n_ctx_train(model)))
        // self.params.seed = seed
        self.params.n_ctx = UInt32(self.maxTokenCount)
        self.params.n_batch = self.params.n_ctx
        self.params.n_threads = processorCount
        self.params.n_threads_batch = processorCount
        self.topK = topK
        self.topP = topP
        self.temp = temp
        self.model = model
        self.history = history
        if let vocab = llama_model_get_vocab(model) {
            self.totalTokenCount = Int(llama_vocab_n_tokens(vocab))
        } else {
            self.totalTokenCount = 0
        }
        self.newlineToken = model.newLineToken
        self.stopSequence = stopSequence?.utf8CString
        self.stopSequenceLength = (self.stopSequence?.count ?? 1) - 1
        self.batch = llama_batch_init(Int32(self.maxTokenCount), 0, 1)

        /// sampler to run with default parameters
        let sparams = llama_sampler_chain_default_params()
        self.sampler = llama_sampler_chain_init(sparams)

        if let sampler = self.sampler {
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(topK))
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(topP, 1))
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp))
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed))
        }
    }

    deinit {
        if let mtmdContext {
            mtmd_free(mtmdContext)
        }
        llama_model_free(self.model)
    }

    public convenience init(
        from url: URL,
        template: Template,
        history: [Chat] = [],
        seed: UInt32 = .random(in: .min ... .max),
        topK: Int32 = 40,
        topP: Float = 0.95,
        temp: Float = 0.8,
        maxTokenCount: Int32 = 2048
    ) {
        self.init(
            from: url.path,
            stopSequence: template.stopSequence,
            history: history,
            seed: seed,
            topK: topK,
            topP: topP,
            temp: temp,
            maxTokenCount: maxTokenCount
        )
        self.preprocess = template.preprocess
        self.template = template
    }

    /// Stops ongoing text generation
    @InferenceActor
    public func stop() {
        guard self.inferenceTask != nil else { return }

        self.inferenceTask?.cancel()
        self.inferenceTask = nil
        self.batch.clear()
    }

    @InferenceActor
    private func predictNextToken() async -> Token {
        /// Ensure context exists; otherwise, return end token
        guard let context = self.context else { return self.model.endToken }

        /// Check if the task has been canceled
        guard !Task.isCancelled else { return self.model.endToken }
        guard self.inferenceTask != nil else { return self.model.endToken }

        /// Ensure the batch is valid
        guard self.batch.n_tokens > 0 else {
            print("Error: Batch is empty or invalid.")
            return model.endToken
        }

        /// Check if the batch size is within limits
        guard self.batch.n_tokens < self.maxTokenCount else {
            print("Error: Batch token limit exceeded.")
            return model.endToken
        }

        guard let sampler = self.sampler else {
            fatalError("Sampler not initialized")
        }

        /// Sample the next token with a valid context
        let token = llama_sampler_sample(sampler, context.pointer, self.batch.n_tokens - 1) // Use batch token count for correct context

        metrics.recordToken()

        self.batch.clear()
        self.batch.add(token, self.nPast, [0], true)
        self.nPast += 1 // Increment the token count after predicting a new token
        context.decode(self.batch)
        return token
    }

    /// Clears conversation history and resets model state
    @InferenceActor
    public func clearHistory() async {
        history.removeAll()
        nPast = 0 /// Reset token count when clearing history
        await setOutput(to: "")
        context = nil
        savedState = nil
        self.batch.clear()
        /// Reset any other state variables if necessary
        /// For example, if you have a variable tracking the current conversation context:
        /// currentContext = nil
    }

    @InferenceActor
    private func tokenizeAndBatchInput(message input: borrowing String) -> Bool {
        guard self.inferenceTask != nil else { return false }
        guard !input.isEmpty else { return false }
        context = context ?? .init(model, params)
        let tokens = encode(input)
        self.inputTokenCount = Int32(tokens.count)

        metrics.inputTokenCount = self.inputTokenCount

        if self.maxTokenCount <= self.nPast + self.inputTokenCount {
            self.trimKvCache()
        }
        for (i, token) in tokens.enumerated() {
            let isLastToken = i == tokens.count - 1

            self.batch.add(token, self.nPast, [0], isLastToken)
            nPast += 1
        }

        /// Check batch has not been cleared by a side effect (stop button) at the time of decoding
        guard self.batch.n_tokens > 0 else { return false }

        self.context.decode(self.batch)
        return true
    }

    @InferenceActor
    private func tokenizeAndBatchInputWithVision(message input: String, image: UIImage, projectorURL: URL?) -> Bool {
        guard self.inferenceTask != nil else { return false }
        guard let projectorURL else { return false }

        context = context ?? .init(model, params)

        guard let mtmdCtx = ensureMtmdContext(projectorURL: projectorURL) else { return false }
        guard let rgb = imageToRGBBytes(image) else { return false }

        let marker = String(cString: mtmd_default_marker())
        let visionPrompt = "\(marker)\n\(input)"
        let processedInput = self.preprocess(visionPrompt, self.history, self)

        let bitmap = rgb.data.withUnsafeBytes { buffer -> mtmd_bitmap? in
            guard let base = buffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return nil }
            return mtmd_bitmap_init(UInt32(rgb.width), UInt32(rgb.height), base)
        }
        guard let bitmap else { return false }
        defer { mtmd_bitmap_free(bitmap) }

        guard let chunks = mtmd_input_chunks_init() else { return false }
        defer { mtmd_input_chunks_free(chunks) }

        let tokenizeResult: Int32 = processedInput.withCString { cPrompt in
            var inputText = mtmd_input_text()
            inputText.text = cPrompt
            inputText.add_special = self.nPast == 0
            inputText.parse_special = true
            var bitmapArray: [mtmd_bitmap?] = [bitmap]
            return bitmapArray.withUnsafeMutableBufferPointer { buffer in
                mtmd_tokenize(mtmdCtx, chunks, &inputText, buffer.baseAddress, 1)
            }
        }
        guard tokenizeResult == 0 else { return false }

        self.batch.clear()
        self.inputTokenCount = 0

        let chunkCount = mtmd_input_chunks_size(chunks)
        for idx in 0..<chunkCount {
            guard let chunk = mtmd_input_chunks_get(chunks, idx) else { continue }
            let chunkType = mtmd_input_chunk_get_type(chunk)
            let chunkTypeRaw = withUnsafeBytes(of: chunkType) { $0.load(as: Int32.self) }
            if chunkTypeRaw == MTMD_INPUT_CHUNK_TYPE_TEXT {
                var nTokens: Int = 0
                guard let tokenPtr = mtmd_input_chunk_get_tokens_text(chunk, &nTokens) else { continue }
                let tokens = Array(UnsafeBufferPointer(start: tokenPtr, count: nTokens))
                self.inputTokenCount += Int32(tokens.count)
                decodeTokens(tokens, logitLast: idx == chunkCount - 1)
            } else if chunkTypeRaw == MTMD_INPUT_CHUNK_TYPE_IMAGE {
                if self.batch.n_tokens > 0 {
                    self.context.decode(self.batch)
                    self.batch.clear()
                }
                guard mtmd_encode_chunk(mtmdCtx, chunk) == 0 else { return false }
                guard let embd = mtmd_get_output_embd(mtmdCtx) else { return false }
                guard decodeImageChunk(mtmdCtx: mtmdCtx, chunk: chunk, embd: embd) else { return false }
            }
        }

        metrics.inputTokenCount = self.inputTokenCount

        if self.batch.n_tokens > 0 {
            self.context.decode(self.batch)
        }
        return true
    }

    private func decodeTokens(_ tokens: [Token], logitLast: Bool) {
        for (index, token) in tokens.enumerated() {
            let isLastToken = logitLast && index == tokens.count - 1
            self.batch.add(token, self.nPast, [0], isLastToken)
            self.nPast += 1
        }
    }

    private func ensureMtmdContext(projectorURL: URL) -> mtmd_context? {
        guard MTMDRuntime.isAvailable else { return nil }
        if let mtmdContext, mtmdProjectorPath == projectorURL.path {
            return mtmdContext
        }

        if let mtmdContext {
            mtmd_free(mtmdContext)
            self.mtmdContext = nil
        }

        guard projectorURL.lastPathComponent.lowercased().contains("mmproj") else { return nil }
        guard FileManager.default.fileExists(atPath: projectorURL.path) else { return nil }
        guard let projectorSize = try? projectorURL.resourceValues(forKeys: [.fileSizeKey]).fileSize,
              projectorSize > 1_000_000 else {
            return nil
        }
        guard let projectorHandle = try? FileHandle(forReadingFrom: projectorURL) else { return nil }
        defer { try? projectorHandle.close() }
        let magic = projectorHandle.readData(ofLength: 4)
        guard String(data: magic, encoding: .utf8) == "GGUF" else { return nil }

        var params = mtmd_context_params_default()
        params.n_threads = max(1, self.params.n_threads)
        if projectorURL.lastPathComponent.lowercased().contains("qwen") {
            params.image_min_tokens = 1024
        }
        guard self.model != nil else { return nil }
        let ctx: mtmd_context? = projectorURL.path.withCString { cPath in
            mtmd_init_from_file(cPath, self.model, params)
        }
        guard let ctx else { return nil }
        self.mtmdContext = ctx
        self.mtmdProjectorPath = projectorURL.path
        return ctx
    }

    private func decodeImageChunk(mtmdCtx: mtmd_context, chunk: mtmd_input_chunk, embd: UnsafeMutablePointer<Float>) -> Bool {
        let nTokens = Int(mtmd_input_chunk_get_n_tokens(chunk))
        let nBatch = max(1, Int(self.params.n_batch))
        let nEmb = Int(llama_model_n_embd_inp(self.model))
        let useMrope = mtmd_decode_use_mrope(mtmdCtx)
        let nPosPerEmbd = useMrope ? 4 : 1
        let seqId: llama_seq_id = 0

        var pos = [llama_pos](repeating: 0, count: nTokens * nPosPerEmbd)
        if useMrope {
            guard let imageTokens = mtmd_input_chunk_get_tokens_image(chunk) else { return false }
            let nx = Int(mtmd_image_tokens_get_nx(imageTokens))
            let ny = Int(mtmd_image_tokens_get_ny(imageTokens))
            for y in 0..<ny {
                for x in 0..<nx {
                    let i = y * nx + x
                    if i >= nTokens { break }
                    pos[i] = self.nPast
                    pos[i + nTokens] = self.nPast + Int32(y)
                    pos[i + nTokens * 2] = self.nPast + Int32(x)
                    pos[i + nTokens * 3] = 0
                }
            }
        } else {
            for i in 0..<nTokens {
                pos[i] = self.nPast + Int32(i)
            }
        }

        let posSnapshot = pos
        var nSeqId = [Int32](repeating: 1, count: nTokens)
        var logits = [Int8](repeating: 0, count: nTokens)
        var seqIdStorage: [llama_seq_id] = [seqId]
        var seqIdPtrs = Array<UnsafeMutablePointer<llama_seq_id>?>(repeating: nil, count: nTokens + 1)

        return seqIdStorage.withUnsafeMutableBufferPointer { seqIdPtr in
            for i in 0..<nTokens {
                seqIdPtrs[i] = seqIdPtr.baseAddress
            }
            seqIdPtrs[nTokens] = nil
            return pos.withUnsafeMutableBufferPointer { posPtr in
                return nSeqId.withUnsafeMutableBufferPointer { nSeqPtr in
                    return seqIdPtrs.withUnsafeMutableBufferPointer { seqPtrs in
                        return logits.withUnsafeMutableBufferPointer { logitsPtr in
                            let useNonCausal = mtmd_decode_use_non_causal(mtmdCtx)
                            if useNonCausal {
                                llama_set_causal_attn(self.context.pointer, false)
                            }
                            defer {
                                if useNonCausal {
                                    llama_set_causal_attn(self.context.pointer, true)
                                }
                            }

                            let nImgBatches = (nTokens + nBatch - 1) / nBatch
                            for batchIndex in 0..<nImgBatches {
                                let offset = batchIndex * nBatch
                                let nTokensBatch = min(nBatch, nTokens - offset)
                                if nTokensBatch <= 0 { continue }

                                let embdPtr = embd.advanced(by: offset * nEmb)
                                if useMrope {
                                    var posView: [llama_pos] = []
                                    posView.reserveCapacity(nTokensBatch * nPosPerEmbd)
                                    for dim in 0..<nPosPerEmbd {
                                        let start = dim * nTokens + offset
                                        let end = start + nTokensBatch
                                        for idx in start..<end {
                                            posView.append(posSnapshot[idx])
                                        }
                                    }
                                    let ok = posView.withUnsafeMutableBufferPointer { posViewPtr -> Bool in
                                        var batch = llama_batch(
                                            n_tokens: Int32(nTokensBatch),
                                            token: nil,
                                            embd: embdPtr,
                                            pos: posViewPtr.baseAddress,
                                            n_seq_id: nSeqPtr.baseAddress!.advanced(by: offset),
                                            seq_id: seqPtrs.baseAddress!.advanced(by: offset),
                                            logits: logitsPtr.baseAddress!.advanced(by: offset)
                                        )
                                        return llama_decode(self.context.pointer, batch) == 0
                                    }
                                    if !ok { return false }
                                } else {
                                    var batch = llama_batch(
                                        n_tokens: Int32(nTokensBatch),
                                        token: nil,
                                        embd: embdPtr,
                                        pos: posPtr.baseAddress!.advanced(by: offset),
                                        n_seq_id: nSeqPtr.baseAddress!.advanced(by: offset),
                                        seq_id: seqPtrs.baseAddress!.advanced(by: offset),
                                        logits: logitsPtr.baseAddress!.advanced(by: offset)
                                    )
                                    if llama_decode(self.context.pointer, batch) != 0 { return false }
                                }
                            }

                            self.nPast += mtmd_input_chunk_get_n_pos(chunk)
                            return true
                        }
                    }
                }
            }
        }
    }

    private func imageToRGBBytes(_ image: UIImage) -> (data: Data, width: Int, height: Int)? {
        guard let cgImage = image.cgImage else { return nil }
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var rgba = [UInt8](repeating: 0, count: bytesPerRow * height)
        guard let context = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        var rgb = [UInt8]()
        rgb.reserveCapacity(width * height * 3)
        for i in stride(from: 0, to: rgba.count, by: 4) {
            rgb.append(rgba[i])
            rgb.append(rgba[i + 1])
            rgb.append(rgba[i + 2])
        }
        return (Data(rgb), width, height)
    }

    /// Decodes a token, checks for the stop sequence, and yields decoded text.
    /// If the complete stop sequence is found, it stops yielding and returns false.
    @InferenceActor
    private func emitDecoded(token: Token, to output: borrowing AsyncStream<String>.Continuation) -> Bool {
        struct saved {
            static var stopSequenceEndIndex = 0
            static var letters: [CChar] = []
        }
        guard self.inferenceTask != nil else { return false }
        guard token != model.endToken else { return false }

        let word = decode(token) /// Decode the token directly

        guard let stopSequence else {
            output.yield(word)
            return true
        }

        /// Existing stop sequence handling logic
        var found = 0 < saved.stopSequenceEndIndex
        var letters: [CChar] = []
        for letter in word.utf8CString {
            guard letter != 0 else { break }
            if letter == stopSequence[saved.stopSequenceEndIndex] {
                saved.stopSequenceEndIndex += 1
                found = true
                saved.letters.append(letter)
                guard saved.stopSequenceEndIndex == stopSequenceLength else { continue }
                saved.stopSequenceEndIndex = 0
                saved.letters.removeAll()
                return false
            } else if found {
                saved.stopSequenceEndIndex = 0
                if !saved.letters.isEmpty {
                    let prefix = String(cString: saved.letters + [0])
                    output.yield(prefix + word)
                    saved.letters.removeAll()
                }
                output.yield(word)
                return true
            }
            letters.append(letter)
        }
        if !letters.isEmpty {
            output.yield(found ? String(cString: letters + [0]) : word)
        }
        return true
    }

    @InferenceActor
    private func generateResponseStream(from input: String) -> AsyncStream<String> {
        AsyncStream<String> { output in
            Task { [weak self] in
                guard let self = self else { return output.finish() } /// Safely unwrap `self`
                /// Use `self` safely now that it's unwrapped

                guard self.inferenceTask != nil else { return output.finish() }

                defer {
                    if !FeatureFlags.useLLMCaching {
                        self.context = nil
                    }
                }

                guard self.tokenizeAndBatchInput(message: input) else {
                    return output.finish()
                }

                metrics.start()
                var token = await self.predictNextToken()
                while self.emitDecoded(token: token, to: output) {
                    if self.nPast >= self.maxTokenCount {
                        self.trimKvCache()
                    }
                    token = await self.predictNextToken()
                }

                metrics.stop()
                output.finish()
            }
        }
    }

    @InferenceActor
    private func generateVisionResponseStream(from input: String, image: UIImage, projectorURL: URL?) -> AsyncStream<String> {
        AsyncStream<String> { output in
            Task { [weak self] in
                guard let self = self else { return output.finish() }
                guard self.inferenceTask != nil else { return output.finish() }

                defer {
                    if !FeatureFlags.useLLMCaching {
                        self.context = nil
                    }
                }

                guard self.tokenizeAndBatchInputWithVision(message: input, image: image, projectorURL: projectorURL) else {
                    return output.finish()
                }

                metrics.start()
                var token = await self.predictNextToken()
                while self.emitDecoded(token: token, to: output) {
                    if self.nPast >= self.maxTokenCount {
                        self.trimKvCache()
                    }
                    token = await self.predictNextToken()
                }

                metrics.stop()
                output.finish()
            }
        }
    }

    /// Halves the llama_kv_cache by removing the oldest half of tokens and shifting the newer half to the beginning.
    /// Updates `nPast` to reflect the reduced cache size.
    @InferenceActor
    private func trimKvCache() {
        let seq_id: Int32 = 0
        let beginning: Int32 = 0
        let middle = Int32(self.maxTokenCount / 2)

        guard let mem = llama_get_memory(self.context.pointer) else { return }

        /// Remove the oldest half
        _ = llama_memory_seq_rm(mem, seq_id, beginning, middle)

        /// Shift the newer half to the start
        llama_memory_seq_add(
            mem,
            seq_id,
            middle,
            Int32(self.maxTokenCount), -middle
        )

        /// Update nPast
        let posMax = llama_memory_seq_pos_max(mem, seq_id)
        self.nPast = posMax + 1
        print("kv cache trimmed: llama_kv_cache(nPast=\(self.nPast))")
    }

    private func getTestLoopbackResponse() -> AsyncStream<String> {
        return AsyncStream { continuation  in
            Task {
                continuation.yield("This is a test loop-back response:\n")
                for i in 0...5 {
                    try? await Task.sleep(nanoseconds: 1_000_000_000 / 2)
                    continuation.yield("\(i) - \(input)\n")
                }
                continuation.finish()
            }
        }
    }

    @InferenceActor
    public func performInference(
        to input: String,
        with makeOutputFrom: @escaping (AsyncStream<String>) async -> String,
        usingVision: (@InferenceActor () -> AsyncStream<String>)? = nil
    ) async {
        self.inferenceTask?.cancel() /// Cancel any ongoing inference task
        self.inferenceTask = Task { [weak self] in
            guard let self = self else { return }

            self.input = input
            let processedInput = self.preprocess(input, self.history, self)
            let responseStream = self.loopBackTestResponse
                ? self.getTestLoopbackResponse()
                : (usingVision?() ?? self.generateResponseStream(from: processedInput))

            /// Generate the output string using the async closure
            let output = (await makeOutputFrom(responseStream)).trimmingCharacters(in: .whitespacesAndNewlines)

            await MainActor.run {
                if !output.isEmpty {
                    /// Update history and process the final output on the main actor
                    self.history.append(Chat(role: .bot, content: output))
                }


                self.postprocess(output)
            }

            self.inputTokenCount = 0
            /// Save the state after generating a response
            if FeatureFlags.useLLMCaching {
                self.savedState = saveState()
            }

            if Task.isCancelled {
                return
            }
        }

        await inferenceTask?.value
    }

    /// Generates a response to the given input
    /// - Parameter input: User input text to respond to
    /// - Note: Updates history and output property with generated response
    open func respond(to input: String) async {
        /// Restore the state before generating a response
        if let savedState = FeatureFlags.useLLMCaching ? self.savedState : nil {
            restoreState(from: savedState)
        }

        await performInference(to: input) { [self] response in
            await setOutput(to: "")
            for await responseDelta in response {
                update(responseDelta)
                await setOutput(to: output + responseDelta)
            }
            update(nil)
            let trimmedOutput = output.trimmingCharacters(in: .whitespacesAndNewlines)


            self.rollbackLastUserInputIfEmptyResponse(trimmedOutput)

            await setOutput(to: trimmedOutput.isEmpty ? "..." : trimmedOutput)
            return output
        }
    }

    /// Generates a response to the given input with an attached image (multimodal path)
    open func respond(to input: String, image: UIImage, projectorURL: URL?) async {
        if let savedState = FeatureFlags.useLLMCaching ? self.savedState : nil {
            restoreState(from: savedState)
        }

        await performInference(to: input) { [self] response in
            await setOutput(to: "")
            for await responseDelta in response {
                update(responseDelta)
                await setOutput(to: output + responseDelta)
            }
            update(nil)
            let trimmedOutput = output.trimmingCharacters(in: .whitespacesAndNewlines)
            self.rollbackLastUserInputIfEmptyResponse(trimmedOutput)
            await setOutput(to: trimmedOutput.isEmpty ? "..." : trimmedOutput)
            return output
        } usingVision: { [self] in
            self.generateVisionResponseStream(from: input, image: image, projectorURL: projectorURL)
        }
    }

    /// If the model fails to produce a response (empty output), remove the last user input’s tokens
    /// from the KV cache to prevent the model’s internal state from being "poisoned" by bad input.
    private func rollbackLastUserInputIfEmptyResponse(_ response: String) {
        if response.isEmpty && self.inputTokenCount > 0 {
            let seq_id = Int32(0)
            let startIndex = self.nPast - self.inputTokenCount
            let endIndex = self.nPast
            if let mem = llama_get_memory(self.context.pointer) {
                _ = llama_memory_seq_rm(mem, seq_id, startIndex, endIndex)
            }
        }
    }

    private func decode(_ token: Token) -> String {
        multibyteCharacter.removeAll(keepingCapacity: true) /// Reset multibyte buffer
        return model.decode(token, with: &multibyteCharacter)
    }

    /// Encodes text into model tokens
    /// - Parameter text: Input text to encode
    /// - Returns: Array of token IDs
    @inlinable
    public func encode(_ text: borrowing String) -> [Token] {
        model.encode(text)
    }
}


extension LLM {
    /// Saves the current model state
    /// - Returns: Data object containing serialized state, or nil if saving fails
    /// - Note: Used for continuing conversations across multiple interactions
    public func saveState() -> Data? {
        /// Ensure the context exists
        guard let contextPointer = self.context?.pointer else {
            print("Error: llama_context pointer is nil.")
            return nil
        }

        /// Get the size of the state
        let stateSize = llama_state_get_size(contextPointer)
        guard stateSize > 0 else {
            print("Error: Unable to retrieve state size.")
            return nil
        }

        /// Allocate a buffer for the state data
        var stateData = Data(count: stateSize)
        stateData.withUnsafeMutableBytes { (pointer: UnsafeMutableRawBufferPointer) in
            if let baseAddress = pointer.baseAddress {
                let bytesWritten = llama_state_get_data(contextPointer, baseAddress.assumingMemoryBound(to: UInt8.self), stateSize)
                assert(bytesWritten == stateSize, "Error: Written state size does not match expected size.")
            }
        }
        return stateData
    }

    /// Restores a previously saved model state
    /// - Parameter stateData: Serialized state data from saveState()
    public func restoreState(from stateData: Data) {
        /// Ensure the context exists
        guard let contextPointer = self.context?.pointer else {
            print("Error: llama_context pointer is nil.")
            return
        }

        /// Set the state data
        stateData.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) in
            if let baseAddress = pointer.baseAddress {
                let bytesRead = llama_state_set_data(contextPointer, baseAddress.assumingMemoryBound(to: UInt8.self), stateData.count)
                assert(bytesRead == stateData.count, "Error: Read state size does not match expected size.")
            }
        }

        let beginningOfSequenceOffset: Int32 = 1
        if let mem = llama_get_memory(self.context.pointer) {
            let posMax = llama_memory_seq_pos_max(mem, 0)
            self.nPast = posMax + beginningOfSequenceOffset
        }
    }
}
