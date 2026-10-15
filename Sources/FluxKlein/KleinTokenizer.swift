import Foundation
import ZImageNative

/// Klein's prompt tokenization as mflux's `Flux2KleinWeightDefinition` configures it: `Qwen2TokenizerFast`, the Qwen3 chat template
/// with `add_generation_prompt` and `enable_thinking: False`, `max_length` 512, right truncation, right padding to `max_length`
/// (`LanguageTokenizer`, padding `max_length`). The byte-level BPE is the one the Z-Image port already implements; the pins here are
/// Klein's own (its `tokenizer_config.json` differs from Z-Image's).
public enum KleinTokenizer {
    public static let maxLength = QwenTokenizer.maxLength

    /// `tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
    /// enable_thinking=False)` with the pinned `chat_template.jinja`: the empty think block follows the assistant header.
    public static func renderChat(prompt: String) -> String {
        "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    public static func load(modelDirectory: URL) throws -> QwenTokenizer {
        try QwenTokenizer.load(modelDirectory: modelDirectory, pinnedFileHashes: KleinPins.tokenizerFiles, label: "FLUX.2 Klein 4B")
    }

    public static func tokenize(_ prompt: String, with tokenizer: QwenTokenizer) -> TokenizedPrompt {
        tokenizer.tokenize(prompt, chat: renderChat(prompt:))
    }
}
