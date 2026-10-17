import FluxKlein
import Foundation

public enum KleinStartingImagePins {
    public static let runtimeModules = KleinGenerationPins.runtimeModules.merging([
        "mflux.models.common.latent_creator.latent_creator": "0ed8f0c34bdbb7a1881b0e9e03c72359e60ac2f012982c440f9c8f11e952760e",
        "mflux.models.flux2.variants.edit.flux2_klein_edit_helpers": "84b54f1d7a9cf735592f3df5341eb1619fe6c6bd7bbc4501969dd94352e01c46",
        "mflux.utils.exif_orientation": "11c808fbc3c066a2d7fa005e312de0796822ac48b3215bab88c376387b2136a5",
    ]) { a, _ in a }
}
