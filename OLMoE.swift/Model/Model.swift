//
//  Model.swift
//  OLMoE.swift
//
//  Created by Ken Adamson on 11/17/24.
//


import Foundation
import llama

extension Model {
    /// Token representing the end of sequence
    public var endToken: Token {
        guard let vocab = llama_model_get_vocab(self) else { return 0 }
        return llama_vocab_eos(vocab)
    }

    /// Token representing a newline character
    public var newLineToken: Token {
        guard let vocab = llama_model_get_vocab(self) else { return 0 }
        return llama_vocab_nl(vocab)
    }

    /// Determines whether Beginning-of-Sequence (BOS) token should be added
    /// - Returns: True if BOS token should be added, based on model vocabulary type
    public func shouldAddBOS() -> Bool {
        guard let vocab = llama_model_get_vocab(self) else { return true }
        let addBOS = llama_vocab_get_add_bos(vocab)
        guard !addBOS else {
            return llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_SPM
        }
        return addBOS
    }

    /// Decodes a single token to string without handling multibyte characters
    /// - Parameter token: The token to decode
    /// - Returns: Decoded string representation of the token
    public func decodeOnly(_ token: Token) -> String {
        var nothing: [CUnsignedChar] = []
        return decode(token, with: &nothing)
    }

    /// Decodes a token to string while handling multibyte characters
    /// - Parameters:
    ///   - token: The token to decode
    ///   - multibyteCharacter: Buffer for handling multibyte character sequences
    /// - Returns: Decoded string representation of the token
    public func decode(_ token: Token, with multibyteCharacter: inout [CUnsignedChar]) -> String {
        guard let vocab = llama_model_get_vocab(self) else { return "" }
        var bufferLength = 16
        var buffer: [CChar] = .init(repeating: 0, count: bufferLength)
        let actualLength = Int(llama_token_to_piece(vocab, token, &buffer, Int32(bufferLength), 0, false))
        guard 0 != actualLength else { return "" }
        if actualLength < 0 {
            bufferLength = -actualLength
            buffer = .init(repeating: 0, count: bufferLength)
            llama_token_to_piece(vocab, token, &buffer, Int32(bufferLength), 0, false)
        } else {
            buffer.removeLast(bufferLength - actualLength)
        }
        if multibyteCharacter.isEmpty, let decoded = String(cString: buffer + [0], encoding: .utf8) {
            return decoded
        }
        multibyteCharacter.append(contentsOf: buffer.map { CUnsignedChar(bitPattern: $0) })
        guard let decoded = String(data: .init(multibyteCharacter), encoding: .utf8) else { return "" }
        multibyteCharacter.removeAll(keepingCapacity: true)
        return decoded
    }

    /// Encodes text into model tokens
    /// - Parameters:
    ///   - text: Input text to encode
    /// - Returns: Array of token IDs representing the encoded text
    /// - Note: Automatically handles BOS token addition and logs the resulting tokens for debugging
    public func encode(_ text: borrowing String) -> [Token] {
        guard let vocab = llama_model_get_vocab(self) else { return [] }
        let addBOS = shouldAddBOS()
        let count = Int32(text.cString(using: .utf8)!.count)
        var tokenCount = count + 1
        let cTokens = UnsafeMutablePointer<llama_token>.allocate(capacity: Int(tokenCount)); defer { cTokens.deallocate() }
        tokenCount = llama_tokenize(vocab, text, count, cTokens, tokenCount, addBOS, false)
        let tokens = (0..<Int(tokenCount)).map { cTokens[$0] }

        print("Encoded tokens: \(tokens)")  // Add this line to log the resulting tokens

        return tokens
    }
}
