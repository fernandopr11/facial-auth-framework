import XCTest
@testable import FacialAuthFramework

final class FacialAuthFrameworkTests: XCTestCase {
    
    func testFrameworkInitialization() {
        // Test que el framework se inicializa correctamente
        let framework = FacialAuthFramework.shared
        XCTAssertNotNil(framework)
    }
    
    func testVersionInfo() {
        // Test de información de versión
        XCTAssertEqual(FacialAuthFramework.version, "1.0.0")
        XCTAssertEqual(FacialAuthFramework.buildNumber, "1")
    }
    
    func testDefaultConfiguration() {
        // Test de configuración por defecto
        let config = FacialAuthConfiguration.default
        XCTAssertEqual(config.securityLevel, .high)
        XCTAssertEqual(config.confidenceThreshold, 0.85)
        XCTAssertFalse(config.allowMultipleFaces)
        XCTAssertEqual(config.cameraTimeout, 10.0)
    }
    
    func testUserProfileCreation() {
        // Test de creación de perfil de usuario
        let user = UserProfile(name: "Fernando")
        XCTAssertEqual(user.name, "Fernando")
        XCTAssertNotNil(user.id)
        XCTAssertTrue(user.dateCreated <= Date())
    }
    
    func testTrueDepthCapabilities() {
        // Test básico de capacidades (funcionará en simulador)
        let deviceModel = TrueDepthCapabilities.deviceModel
        XCTAssertFalse(deviceModel.isEmpty)
        print("🔍 Device Model: \(deviceModel)")
    }
    
    func testErrorDescriptions() {
        // Test de descripciones de errores
        let error = FacialAuthError.cameraNotAvailable
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("Camera"))
    }
}
