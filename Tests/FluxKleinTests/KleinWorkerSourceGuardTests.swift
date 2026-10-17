import Darwin
import Foundation
import XCTest
@testable import FluxKlein

/// Slice 5 source guards: the host's exact strength expression, and a fresh-directory creation that cannot reuse an existing leaf.
final class KleinHostStrengthTests: XCTestCase {
    /// `StartingImageStrength.imageStrength` in the app's StartingImage.swift, restated verbatim.
    private func host(_ steps: Int, _ change: Int) -> Double { Double(steps - change + 1) / Double(steps) - 0.5 / Double(steps) }

    func testAll28HostPairsMapToTheHostsExactBitPattern() throws {
        var count = 0
        for steps in 2 ... 8 {
            for change in 1 ..< steps {
                let strength = try KleinStartingImageRequest.strength(steps: steps, change: change)
                XCTAssertEqual(strength.bitPattern, host(steps, change).bitPattern, "steps \(steps), change \(change)")
                XCTAssertEqual(try KleinStartingImageRequest.start(steps: steps, strength: strength), steps - change)
                count += 1
            }
        }
        XCTAssertEqual(count, 28)
        // The four frozen job strengths are written as literals and do not go through this mapping.
        XCTAssertEqual(try KleinStartingImageRequest.strength(steps: 4, change: 2), 0.625)
        XCTAssertEqual(try KleinStartingImageRequest.strength(steps: 8, change: 7), 0.1875)
        XCTAssertEqual(try KleinStartingImageRequest.strength(steps: 8, change: 1), 0.9375)
        XCTAssertEqual(try KleinStartingImageRequest.strength(steps: 2, change: 1), 0.75)
        for (steps, change) in [(1, 1), (9, 1), (4, 0), (4, 4), (4, -1)] {
            XCTAssertThrowsError(try KleinStartingImageRequest.strength(steps: steps, change: change))
        }
    }
}

final class KleinFreshDirectoryTests: XCTestCase {
    private func temp() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("klein-fresh-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        let path = try XCTUnwrap(url.path.withCString { Darwin.realpath($0, nil) })
        defer { free(path) }
        return URL(fileURLWithPath: String(cString: path), isDirectory: true)
    }

    func testCreatesMissingParentsAndANewLeafOnly() throws {
        let dir = try temp()
        let leaf = dir.appendingPathComponent("a/b/run")
        try KleinStartingImagePaths.createNewDirectory(leaf)
        var isDirectory: ObjCBool = false
        XCTAssertTrue(FileManager.default.fileExists(atPath: leaf.path, isDirectory: &isDirectory)); XCTAssertTrue(isDirectory.boolValue)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: leaf.path), [])
    }

    func testAnExistingDirectoryIsRejectedAndItsContentsSurvive() throws {
        let dir = try temp()
        let existing = dir.appendingPathComponent("existing")
        try FileManager.default.createDirectory(at: existing, withIntermediateDirectories: false)
        let evidence = existing.appendingPathComponent("evidence.txt")
        try Data("keep".utf8).write(to: evidence)
        // `FileManager.createDirectory(withIntermediateDirectories: true)` would have accepted this leaf silently.
        XCTAssertThrowsError(try KleinStartingImagePaths.createNewDirectory(existing))
        XCTAssertEqual(try String(contentsOf: evidence, encoding: .utf8), "keep")
        XCTAssertThrowsError(try KleinStartingImagePaths.createNewDirectory(dir), "the parent itself exists")
        // A leaf that appears between the check and the mkdir: the POSIX mkdir reports EEXIST instead of reusing it.
        XCTAssertEqual(mkdir(existing.path, 0o755), -1); XCTAssertEqual(errno, EEXIST)
    }

    func testSymlinkedPathsAndModelNestingAreRejectedBeforeAnythingIsCreated() throws {
        let dir = try temp()
        let model = dir.appendingPathComponent("ModelFolder")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: false)
        XCTAssertThrowsError(try KleinStartingImagePaths.createNewDirectory(model.appendingPathComponent("runs/r1"), model: model))
        XCTAssertFalse(FileManager.default.fileExists(atPath: model.appendingPathComponent("runs").path), "rejected before any parent is made")
        let link = dir.appendingPathComponent("link")
        try FileManager.default.createSymbolicLink(atPath: link.path, withDestinationPath: dir.appendingPathComponent("absent").path)
        XCTAssertThrowsError(try KleinStartingImagePaths.createNewDirectory(link.appendingPathComponent("run")))
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent("absent").path))
        // `URL(fileURLWithPath:)` would resolve against the cwd and be absolute; only a scheme-less URL is genuinely relative.
        let relative = try XCTUnwrap(URL(string: "relative/run"))
        XCTAssertFalse(relative.path.hasPrefix("/"))
        XCTAssertThrowsError(try KleinStartingImagePaths.createNewDirectory(relative))
    }
}
