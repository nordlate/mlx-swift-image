import Foundation
import MLX
import MLXImageCore

/// What `Flux2PromptEncoder.encode_prompt` produces for one prompt, plus the intermediate states it is assembled from.
public struct KleinPromptEncoding {
    /// `hidden_states_list[0]`, `[9]`, `[18]`, `[27]`: the embedding output and the selected layer outputs, `[1, 512, 2560]` bfloat16.
    public let states: [Int: MLXArray]
    /// `[1, 512, 7680]` bfloat16: the selected states concatenated layer-major per token (padded positions included).
    public let promptEmbeds: MLXArray
    /// `[1, 512, 4]` int32: `(t, h, w, l)` coordinates with `l = arange(512)`.
    public let textIDs: MLXArray
}

public enum KleinPromptEncoder {
    public static let sequenceLength = KleinTokenizer.maxLength
    public static let embeddingWidth = KleinTextEncoder.selectedLayers.count * TextEncoderLayout.hiddenSize

    /// `Flux2PromptEncoder.prepare_text_ids(prompt_embeds)` with no `t_coord`: per batch row, `stack([t=0, h=0, w=0, arange(S)], axis=1)`.
    public static func textIDs(batch: Int, seqLen: Int) -> MLXArray {
        var rows: [MLXArray] = []
        for _ in 0..<batch {
            let t = MLXArray.zeros([seqLen], dtype: .int32)
            let h = MLXArray.zeros([seqLen], dtype: .int32)
            let w = MLXArray.zeros([seqLen], dtype: .int32)
            let tokenIDs = KleinTextEncoder.arange32(seqLen)
            rows.append(stacked([t, h, w, tokenIDs], axis: 1))
        }
        return stacked(rows, axis: 0)
    }

    /// Encodes one tokenized prompt (a single row, as production generation does): a padded `[1, 512]` int64 batch, or the `[1, 0]` int32 batch
    /// mflux's tokenizer returns for the empty prompt. Production MFLUX runs the empty batch through the same MLX operations and gets
    /// zero-length states (`[1, 0, 2560]`), embeddings (`[1, 0, 7680]`) and text IDs (`[1, 0, 4]`); so does this, with no special path.
    public static func encode(_ tokens: TokenizedPrompt, with encoder: KleinTextEncoder) throws -> KleinPromptEncoding {
        let isEmpty = tokens.inputIDs.isEmpty
        guard tokens.shape == [1, isEmpty ? 0 : sequenceLength], tokens.attentionMask.count == tokens.inputIDs.count,
              tokens.dtype == (isEmpty ? "mlx.core.int32" : "mlx.core.int64") else {
            throw KleinError.invalidInput("tokenized prompt is \(tokens.dtype) \(tokens.shape), expected int64 [1, \(sequenceLength)] or the empty int32 [1, 0]")
        }
        let tokenDType: DType = isEmpty ? .int32 : .int64
        let ids = KleinTextEncoder.tokenArray(tokens.inputIDs, shape: tokens.shape, dtype: tokenDType)
        let mask = KleinTextEncoder.tokenArray(tokens.attentionMask, shape: tokens.shape, dtype: tokenDType)
        let selected = KleinTextEncoder.selectedLayers
        let states = encoder.hiddenStates(inputIDs: ids, attentionMask: mask, indices: Set([0] + selected))
        let promptEmbeds = KleinTextEncoder.assemble(selected.map { states[$0]! })
        let ids4 = textIDs(batch: promptEmbeds.dim(0), seqLen: promptEmbeds.dim(1))
        eval(Array(states.values) + [promptEmbeds, ids4])
        return KleinPromptEncoding(states: states, promptEmbeds: promptEmbeds, textIDs: ids4)
    }
}
