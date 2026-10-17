import FluxKlein
import Foundation

/// What the transformer parity run pins about the snapshot and the production runtime: the transformer files and the further audited
/// MFLUX modules (`reference-code-audit.json`, 21 modules, 2 of them already pinned by `KleinPins`).
public extension KleinTransformerPins {
    public static let transformerFiles: [String: String] = [
        "transformer/0.safetensors": "36796395e3946496f3e118f26fe22594b1d7d871342a59dd7824349d91c08266",
        "transformer/1.safetensors": "9730ae148ca4f5ffc62860748b849a70987f0b7301588248bd98a2af9bc15070",
        "transformer/model.safetensors.index.json": "dbdecbf3a957b37d15bd943b956dbb25aa648a01df363e7dd5ebbe5ce941358d",
    ]

    /// The text-encoding files of slice 1 plus the transformer files: all nine are hashed before and after every run on both sides.
    public static let modelFiles: [String: String] = KleinPins.modelFiles.merging(transformerFiles) { first, _ in first }

    /// SHA-256 of the runtime `.pyc` of the 19 additional audited modules. (`flux2_weight_definition` and `flux2_initializer` are
    /// already in `KleinPins.runtimeModules` with the same hashes.)
    public static let additionalRuntimeModules: [String: String] = [
        "mflux.models.flux2.model.flux2_transformer.transformer": "37588a8332272d5b44603c676abf2ff73669dc442eadbdae90c81aae24ca23a5",
        "mflux.models.flux2.model.flux2_transformer.transformer_block": "bbbc04e60594ecb600235656bd5d180ba672471b20e782abac02cd5013f84dbe",
        "mflux.models.flux2.model.flux2_transformer.single_transformer_block": "5d53a14c85b7f2c2583d311cf8fa844d83d51d9f63ef68a4e9fa15b045a9c5db",
        "mflux.models.flux2.model.flux2_transformer.attention": "dbe6aa3c63d6efcad441f19882103c3eaba37937b08eb23f31694ed768124f15",
        "mflux.models.flux2.model.flux2_transformer.parallel_self_attention": "3817a5577bc775d8f87347fc1cce2c55d9892a8a15c75bfd354638bd9c03f59b",
        "mflux.models.flux2.model.flux2_transformer.feed_forward": "8524e0cbf39196a75244e3b8f81f2c21fd824eb31b3f53a0c9c057744ba811ee",
        "mflux.models.flux2.model.flux2_transformer.modulation": "6a21b3d8f4a2dc8caf77895a2f76bd9763ca8715a00ece2c21c724e269864171",
        "mflux.models.flux2.model.flux2_transformer.pos_embed": "16f46b02e5a124c6afd159e7bdbefdaee607b1dd62a755e84f2c6da46035b4f4",
        "mflux.models.flux2.model.flux2_transformer.timestep_guidance_embeddings": "c8a158645d25981415ca02a7c66287355ba1b2260e26a0a1d1877e3d68cbbf6c",
        "mflux.models.flux2.model.flux2_transformer.flux2_kv_cache": "501702a16048bfc5a1e738bd1130afe517730c317b10283bca0d1e3a5ba52590",
        "mflux.models.flux2.latent_creator.flux2_latent_creator": "86213ee5edbd6c57a09239374c26a4cd130c3cdab0b58a919ac45252a4c7acd9",
        "mflux.models.common.schedulers.flow_match_euler_discrete_scheduler": "2580dd978a812c06458ad08557e3a7efb964a2081698024e581bf41ee068a801",
        "mflux.models.common.config.config": "b366aa09964df211f8f354e2afcff68556b0441e24294a696346909ce418b959",
        "mflux.models.common.config.model_config": "d9d9565ca8b70f8eebe2597e445561351dc3717f981ac10a756795e6c217f5f8",
        "mflux.models.flux.model.flux_transformer.common.attention_utils": "d4e76aaf3634479bb49c0155e65cc95629b7ca6ea1d12dbd305b3f7a3303c401",
        "mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous": "5413b24730ad1064bbd68485ec04c76fa5d283300c04cf6be2f58234ffeed199",
        "mflux.models.flux2.variants.txt2img.flux2_klein": "8aa6b42b6d18d5a66916755205d355cc04a6d1941aec4ae8827feb17d4a68b9a",
        "mflux.models.common.weights.loading.weight_applier": "27daa18080309cf9e3c98147f467c6f330dda7dd2dbc69782e1e5f51b1c5ad46",
        "mflux.models.common.weights.loading.weight_loader": "b0a2a7513a4ab0c0f2f54f8ae1aae086de1a59162092d9c33e95e1f19f21bd24",
    ]

    /// All 28 modules the slice-2 reference imports and hashes: slice 1's nine and the 19 above.
    public static let runtimeModules: [String: String] = KleinPins.runtimeModules.merging(additionalRuntimeModules) { first, _ in first }

    /// Hashes the nine pinned model files and throws, naming the file, when one differs from its pin.
    public static func verifyModelFiles(modelDirectory: URL) throws -> [String: String] {
        var result: [String: String] = [:]
        for (path, expected) in modelFiles.sorted(by: { $0.key < $1.key }) {
            let digest = try KleinHashing.sha256(of: modelDirectory.appendingPathComponent(path))
            result[path] = digest
            guard digest == expected else {
                throw KleinTransformerError.artifactChanged("\(path): SHA-256 \(digest) differs from the pinned \(expected)")
            }
        }
        return result
    }
}
