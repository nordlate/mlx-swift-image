import MLX
import XCTest
@testable import FluxKlein
@testable import MLXImageParity

final class KleinScheduleSanityTests: XCTestCase {
    func testProductionSchedulerPassesTheHarnessSanityCheckForAllOfferedSteps() throws {
        try Device.withDefaultDevice(.cpu) {
            for n in 1...8 {
                let scheduler = try KleinScheduler(steps: n, height: 32, width: 48)
                XCTAssertEqual(KleinTransformerCompare.scheduleFailures(side: "test", label: "n\(n)", sigmas: scheduler.sigmas, timesteps: scheduler.timesteps), [])
            }
        }
    }
}
