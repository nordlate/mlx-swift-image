import Foundation

public enum KleinGenerationPins {
    public static let vaeFiles: [String: String] = [
        "vae/0.safetensors": "688ae0c3d2e93573ce8a6d3561d793cf6de5a4207b2e128cad6cd0d71e43b10f",
        "vae/model.safetensors.index.json": "e13e551dabf8c088c8b3c6f52fca3f9e64ead5c42dd592586c7a1f868607afa4",
    ]
    public static let modelFiles = KleinTransformerPins.modelFiles.merging(vaeFiles) { a, _ in a }
    public static let modelSizes: [String: Int] = [
        "tokenizer/chat_template.jinja": 4168,
        "tokenizer/tokenizer.json": 11422650,
        "tokenizer/tokenizer_config.json": 376,
        "text_encoder/0.safetensors": 2135435237,
        "text_encoder/1.safetensors": 127582075,
        "text_encoder/model.safetensors.index.json": 51332,
        "transformer/0.safetensors": 2145323769,
        "transformer/1.safetensors": 34727728,
        "transformer/model.safetensors.index.json": 26908,
        "vae/0.safetensors": 165107888,
        "vae/model.safetensors.index.json": 17547,
    ]
    public static let additionalRuntimeModules: [String: String] = [
        "mflux.models.flux2.model.flux2_vae.vae": "d2bc407e5e6b713565da8392ba04dd94a9d6ef71f5ad6c4b5a102eac0c0a64eb",
        "mflux.models.flux2.model.flux2_vae.common.batch_norm_stats": "375035da461e5f67eaefa2533599010439e90645a2184b58ac36dc1088e6985b",
        "mflux.models.flux2.model.flux2_vae.common.attention": "dfa37c13053d5326ea0b5b1ab7e0a9ea967ff9e3b5ab0b97ab0433a0b7f3f889",
        "mflux.models.flux2.model.flux2_vae.common.resnet_block_2d": "20313ce63c4f4c253dfd4085e07292860544f70eca28bd408198dc1584bb14a0",
        "mflux.models.flux2.model.flux2_vae.common.unet_mid_block": "c012a43ae0e35c1b8ec4927700fa88a4442563ebccae994a8808df49a851fe64",
        "mflux.models.flux2.model.flux2_vae.common.upsample_2d": "ec579858cf3c0286e59874713de94a805e53de0e28023d332597c9a134c95267",
        "mflux.models.flux2.model.flux2_vae.common.downsample_2d": "1aa65822d8891310666e779cf9a6ff052b215ab73ad6a04d894553b32cffb6a4",
        "mflux.models.flux2.model.flux2_vae.decoder.decoder": "832b697ea4b0bfd65aa5b0b69a253c7720a714d6970bac919c5d31d5afb4645e",
        "mflux.models.flux2.model.flux2_vae.decoder.conv_in": "5b06e69f7c7b9e7bdcb8b449856b3b1275285fb69d51d2c484b6bd09adc8ed8c",
        "mflux.models.flux2.model.flux2_vae.decoder.conv_norm_out": "6b4ba054e7710e42f99eaf39a69d0b4b1e5b36e8303ce063893975de69b4053e",
        "mflux.models.flux2.model.flux2_vae.decoder.conv_out": "1881f38252c11d790f12d07d372afff5b2ed887659baf89deda283423bfbe45f",
        "mflux.models.flux2.model.flux2_vae.decoder.up_decoder_block": "6b4ba077dbe5dee978e82ae3e892b049af94b13091882af4ab3077e17833e071",
        "mflux.models.flux2.model.flux2_vae.encoder.encoder": "f9b99b7826415b03d47e711e5366ace3e7e3053f41bbe3ac63ee8f11585af063",
        "mflux.models.flux2.model.flux2_vae.encoder.conv_in": "8e882d401dffeb9eed2a4781e860e262147a069ad603487d67230ecc05ae2909",
        "mflux.models.flux2.model.flux2_vae.encoder.conv_norm_out": "e6706e21768a77dcebb23fa1fb3e72636c40882da1de56b90e0ee0c71b42e636",
        "mflux.models.flux2.model.flux2_vae.encoder.conv_out": "597f9aad4c7d0fd141788262900e47fab2c57682d6aea28d463262100656f3b1",
        "mflux.models.flux2.model.flux2_vae.encoder.down_encoder_block": "e540516575a20d8ab8ad5ce3a99b2ef9d5eff11038fa3b51b54da4cc3ebe2426",
        "mflux.models.common.vae.vae_util": "0a5930c02098a5b02aff36df455d7395afc5872ea81ebce5219df7e5a365e6fc",
        "mflux.utils.image_util": "1ad981c9295a99dea06dee417ccbde7ec8b117acb17367f68bf2203602ee88e7",
        "mflux.utils.generated_image": "d3dd6763be042971425afc98044a53ccad463d2d52d1e4b324ac92af8c73db7b",
        "mflux.callbacks.callback_manager": "3118542ea99657b402bded15e741ef4139ce2237a86ded95d0bed556dd453c0b",
        "mflux.callbacks.callback_registry": "200febd144d26f2d1b2038f4994adff8a288a4804d7d7e7c9805ecfd6af3c5e6",
        "mflux.callbacks.generation_context": "be6d0dd670fb180879652e864801c4e66a5963e25bcb834f74103275be420173",
        "mflux.callbacks.callback": "7e16cf537f2c361360be40b1d108945e61cb0335b45f6e124077777d0b588e13",
        "mflux.models.common.resolution.config_resolution": "e72a84c41c5143ecd47df4b24a67a8e687d5d78e2124de5d1fc778b0c8d8d47e",
    ]
    public static let runtimeModules = KleinTransformerPins.runtimeModules.merging(additionalRuntimeModules) { a, _ in a }
    public static func verifyModelFiles(modelDirectory: URL) throws -> [String: String] {
        var result: [String: String] = [:]
        for (path, expected) in modelFiles.sorted(by: { $0.key < $1.key }) {
            let url = modelDirectory.appendingPathComponent(path)
            let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize
            guard size == modelSizes[path] else { throw KleinTransformerError.artifactChanged("\(path): file size differs") }
            let digest = try KleinHashing.sha256(of: url)
            guard digest == expected else { throw KleinTransformerError.artifactChanged("\(path): SHA-256 differs") }
            result[path] = digest
        }
        return result
    }
}
