import Foundation

/// Fixed values the converted snapshot contains that do not come from the official weights.
enum KleinConversionConstants {
    /// `rotary_emb.inv_freq` of mflux's Qwen3 text encoder, `1 / 1e6 ** (arange(0, 128, 2) / 128)` in float32 as MLX evaluates it on
    /// the GPU when `mflux-save` builds the model; float32 bit patterns, little-endian when stored. The same for every Klein size.
    static let qwen3InvFreqBits: [UInt32] = [
        0x3F800000, 0x3F4E4BAD, 0x3F263DE0, 0x3F05F6EF, 0x3ED7E89B, 0x3EADFCFE, 0x3E8C3504, 0x3E61F835,
        0x3E361887, 0x3E12BD91, 0x3DEC7FD4, 0x3DBE94C6, 0x3D99940D, 0x3D778511, 0x3D47763E, 0x3D20BC1D,
        0x3D0186E3, 0x3CD0C1A7, 0x3CA8398A, 0x3C879007, 0x3C5A7BF2, 0x3C301051, 0x3C0DE12D, 0x3BE4AA46,
        0x3BB8449C, 0x3B947DAD, 0x3B6F520C, 0x3B40DAC5, 0x3B1B690D, 0x3AFA78F0, 0x3AC9D75C, 0x3AA2A6F6,
        0x3A83126F, 0x3A533F28, 0x3A2A3B43, 0x3A092E02, 0x39DD1725, 0x39B229FA, 0x398F9272, 0x39676490,
        0x393A7753, 0x39164323, 0x38F22CE2, 0x38C327B4, 0x389D43A4, 0x387D75D4, 0x384C3FBC, 0x382497AA,
        0x3804A2B2, 0x37D5C441, 0x37AC431D, 0x378AD0ED, 0x375FBA4F, 0x37344A0D, 0x371148E3, 0x36EA2732,
        0x36BCB0C1, 0x36980E01, 0x36751070, 0x36457BAB, 0x361F23E3, 0x36003DEC, 0x35CEAF78, 0x35A68E4B,
    ]

    static var qwen3InvFreq: [UInt8] { qwen3InvFreqBits.flatMap { withUnsafeBytes(of: $0.littleEndian, Array.init) } }

    /// The `tokenizer_config.json` that `mflux-save` 0.20 writes (`Qwen2TokenizerFast.save_pretrained` under transformers 5); its
    /// SHA-256 is pinned in `KleinPins.tokenizerFiles`. Ends with a newline (the blank line before the closing delimiter).
    static let tokenizerConfig = """
{
  "add_prefix_space": false,
  "backend": "tokenizers",
  "bos_token": null,
  "clean_up_tokenization_spaces": false,
  "eos_token": "<|im_end|>",
  "errors": "replace",
  "is_local": true,
  "local_files_only": true,
  "model_max_length": 131072,
  "pad_token": "<|endoftext|>",
  "split_special_tokens": false,
  "tokenizer_class": "Qwen2Tokenizer",
  "unk_token": null
}

"""
}
