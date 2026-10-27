import XCTest
@testable import FluxKlein
@testable import MLXImageCore
@testable import MLXImageParity

final class KleinPinsTests: XCTestCase {
    func testPinSetsAreComplete() {
        XCTAssertEqual(KleinPins.runtimeModules.count, 9)
        XCTAssertEqual(Set(KleinPins.modelFiles.keys).count, 6)
        XCTAssertTrue(KleinPins.modelFiles.values.allSatisfy { $0.count == 64 })
        XCTAssertTrue(KleinPins.runtimeModules.values.allSatisfy { $0.count == 64 })
        XCTAssertEqual(KleinTransformerPins.modelFiles.count, 9)
        XCTAssertEqual(KleinGenerationPins.modelFiles.count,11)
        XCTAssertEqual(KleinGenerationPins.runtimeModules.count,53)
        XCTAssertEqual(KleinTransformerPins.runtimeModules.count, 28)
        XCTAssertEqual(KleinPins.mlxSwiftRevision, "19601207e9a0de51e03ee6ec0c3c5f3784275075")
    }

    func testOutputContainmentIsByPathComponentNotStringPrefix() {
        let model = URL(fileURLWithPath: "/nonexistent-klein/models/flux2-klein-4b-q4", isDirectory: true)
        XCTAssertTrue(KleinNativeExport.isInside(model.appendingPathComponent("text_encoder"), model))
        XCTAssertTrue(KleinNativeExport.isInside(model, model))
        XCTAssertFalse(KleinNativeExport.isInside(URL(fileURLWithPath: "/nonexistent-klein/models/flux2-klein-4b-q4-out"), model), "a sibling sharing the name prefix")
    }

    func testChangedArtifactIsNamedAndRejected() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("klein-pins-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        for rel in KleinPins.modelFiles.keys {
            let url = dir.appendingPathComponent(rel)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("not the pinned bytes".utf8).write(to: url)
        }
        XCTAssertThrowsError(try KleinHashing.verifyModelFiles(modelDirectory: dir)) { error in
            XCTAssertTrue("\(error)".contains("differs from the pinned"), "\(error)")
        }
        XCTAssertThrowsError(try KleinTokenizer.load(modelDirectory: dir), "a tokenizer directory with other files must be refused")
    }
}

/// (The audit counter is scoped to the measuring task, so this test is safe under parallel test execution.)
/// The production load path hashes no weights (slice 11: ~1.8 s of every Klein load); the harness digests stay available on demand and equal
/// the weights' own `digests()`.
final class KleinLoadDigestTests: XCTestCase {
    func testProductionInitComputesNoDigestsAndHarnessDigestsAreOnDemandAndEqual() throws {
        let model = try KleinTestSupport.requireModel()
        let (generator, loadDigested) = try KleinDigestAudit.measure { try KleinGenerator(modelDirectory: model) }
        XCTAssertEqual(loadDigested, 0, "KleinGenerator.init hashed weights")
        let ((text, transformer, vae), onDemand) = KleinDigestAudit.measure { (generator.weights.text.digests(), generator.weights.transformer.digests(), generator.weights.vae.digests()) }
        XCTAssertEqual(onDemand, text.count + transformer.count + vae.count)
        XCTAssertEqual([text.count, transformer.count, vae.count], [905, KleinTransformerLayout.klein4B.tensorCount, KleinVAELayout.tensorCount])
        // Same tensors, same bytes, same SHA-256 definition as a direct load of the weights.
        XCTAssertEqual(text, try KleinTextWeights.load(modelDirectory: model).digests())
        XCTAssertEqual(transformer, try KleinTransformerWeights.load(modelDirectory: model).digests())
        XCTAssertEqual(vae, try KleinVAEWeights.load(modelDirectory: model).digests())
    }
}
