import Foundation

/// The files of an official FLUX.2 Klein diffusers release on Hugging Face, as `KleinConverter` accepts them. Sizes and SHA-256s
/// were taken from the Hugging Face API tree (the LFS `sha256` of the large files) and, for the small files, computed from a download
/// whose git blob ids matched the tree.
public struct KleinOfficialRelease: Sendable {
    public struct File: Equatable, Sendable {
        /// Path relative to the snapshot folder.
        public let path: String
        public let byteCount: Int64
        public let sha256: String
    }

    public let variant: KleinVariant
    /// Hugging Face repository and the commit the pins were taken from.
    public let repository: String
    public let revision: String
    /// Files the conversion reads; each must be present with this size and SHA-256.
    public let requiredFiles: [File]
    /// Files of the release the conversion does not need. They may be present or absent and are not hashed.
    public let optionalFiles: [String]

    /// The release `KleinConverter` can convert for a variant; nil when the variant has no conversion pins.
    public static func release(for variant: KleinVariant) -> KleinOfficialRelease? {
        switch variant {
        case .klein9B: klein9B
        case .klein4B: nil
        }
    }

    /// The required files that are safetensors weights (hashed while they are converted).
    public var weightFiles: [File] { requiredFiles.filter { $0.path.hasSuffix(".safetensors") } }

    /// `black-forest-labs/FLUX.2-klein-9B` (FLUX Non-Commercial License), 15 required files, 34.5 GB of weights.
    public static let klein9B = KleinOfficialRelease(
        variant: .klein9B,
        repository: "black-forest-labs/FLUX.2-klein-9B",
        revision: "92196c8e11f7b6cf2b7493e037d8c5345c559216",
        requiredFiles: [
            File(path: "model_index.json", byteCount: 446, sha256: "51a76cb1cf3ed37423a1128c79c22faee8e6fbe7f5aaeb737f0a258930dbaac0"),
            File(path: "transformer/config.json", byteCount: 542, sha256: "e82d0d325aff03c3b3b33a1634c47a5f88867478f53071b9c9a39c99010c5d46"),
            File(path: "transformer/diffusion_pytorch_model.safetensors.index.json", byteCount: 24875,
                 sha256: "96fe598b12bdfa266f54c2a6027dedc6e87016cce7e19cb2821a8b7deaf9deb9"),
            File(path: "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", byteCount: 9_801_069_272,
                 sha256: "cb942a7072865a1d06e47a3361f9ba8746e68ad207c8499083bcb735869f5102"),
            File(path: "transformer/diffusion_pytorch_model-00002-of-00002.safetensors", byteCount: 8_356_121_608,
                 sha256: "ca568a31d19c03ddbcfd8b2d4ec7dbd16dcefbaa50b7aef1b8ceefd6e6eb0970"),
            File(path: "text_encoder/config.json", byteCount: 1538, sha256: "57866e90a1d6328a7ed53eca732bce106f86c76ab99d7629e01f0a319fa57998"),
            File(path: "text_encoder/model.safetensors.index.json", byteCount: 32914,
                 sha256: "e7e4bdc58d3302c97357f27979b270987dd620cf0cd9b1c130e3a51c9d64df95"),
            File(path: "text_encoder/model-00001-of-00004.safetensors", byteCount: 4_902_257_696,
                 sha256: "c0dc64934ae0f730ddc80d99af44968d01a89e8454df07d762096ea1356446bc"),
            File(path: "text_encoder/model-00002-of-00004.safetensors", byteCount: 4_915_960_368,
                 sha256: "d58533b468c31caa0222540a8aefddea1d74dc6e1fee928da8556d3d85729d6e"),
            File(path: "text_encoder/model-00003-of-00004.safetensors", byteCount: 4_983_068_496,
                 sha256: "74927fec432e050365bf757bb30348f560a44394efac89f492680b7c910b64fd"),
            File(path: "text_encoder/model-00004-of-00004.safetensors", byteCount: 1_580_230_264,
                 sha256: "cb73f466fade5716702bda38d4e3b321c9358c39889e46fb9d613fb038bfcb2f"),
            File(path: "vae/config.json", byteCount: 821, sha256: "0d6dfb69ae95a5e2ac9836284bbb63d8b38ce67b25ba2dff380752b2a10ab948"),
            File(path: "vae/diffusion_pytorch_model.safetensors", byteCount: 168_120_878,
                 sha256: "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04"),
            File(path: "tokenizer/tokenizer.json", byteCount: 11_422_654, sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
            File(path: "tokenizer/chat_template.jinja", byteCount: 4168, sha256: "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"),
        ],
        optionalFiles: [
            "LICENSE.md", "README.md", ".gitattributes", "editing.jpg", "others.jpg", "realism.jpg", "flux-2-klein-9b.safetensors",
            "scheduler/scheduler_config.json", "text_encoder/generation_config.json",
            "tokenizer/added_tokens.json", "tokenizer/special_tokens_map.json", "tokenizer/tokenizer_config.json", "tokenizer/vocab.json",
            "tokenizer/merges.txt",
        ]
    )
}
