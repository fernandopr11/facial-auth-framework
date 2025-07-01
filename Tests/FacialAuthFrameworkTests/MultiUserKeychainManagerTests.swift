import XCTest
@testable import FacialAuthFramework

final class MultiUserKeychainManagerTests: XCTestCase {
    
    var keychainManager: MultiUserKeychainManager!
    var embeddingManager: SecureEmbeddingManager!
    
    override func setUp() {
        super.setUp()
        
        // Usar manager normal pero con política menos restrictiva para tests
        keychainManager = MultiUserKeychainManager(
            biometricPolicy: .deviceOwnerAuthentication,
            appIdentifier: "test.facial.auth.framework"
        )
        
        do {
            embeddingManager = try SecureEmbeddingManager()
        } catch {
            XCTFail("Failed to initialize SecureEmbeddingManager: \(error)")
        }
    }
    
    override func tearDown() {
        // Cleanup simple sin async
        keychainManager = nil
        embeddingManager = nil
        super.tearDown()
    }
    
    func testUserProfileMetadata() throws {
        // Test simple sin Keychain para verificar que la base funciona
        let userID = "metadata_test"
        let userName = "Metadata Test User"
        let embeddingCount = 5
        
        let metadata = UserProfileMetadata(
            userID: userID,
            name: userName,
            embeddingCount: embeddingCount
        )
        
        XCTAssertEqual(metadata.userID, userID)
        XCTAssertEqual(metadata.name, userName)
        XCTAssertEqual(metadata.embeddingCount, embeddingCount)
        XCTAssertEqual(metadata.version, FacialAuthFramework.version)
        XCTAssertTrue(metadata.dateCreated <= Date())
        XCTAssertTrue(metadata.lastUpdated <= Date())
    }
    
    func testEmbeddingEncryption() throws {
        // Test de encriptación sin usar Keychain
        let embedding: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        let encrypted = try embeddingManager.encrypt(embedding: embedding)
        let decrypted = try embeddingManager.decrypt(encryptedEmbedding: encrypted)
        
        XCTAssertEqual(decrypted, embedding)
    }
    
    func testKeychainManagerInitialization() {
        // Test que el manager se inicializa correctamente
        XCTAssertNotNil(keychainManager)
        // No accedemos a propiedades privadas, solo verificamos que existe
    }
    
    // MARK: - Tests de integración real (comentados para evitar errores de simulador)
    
    /*
    func testStoreAndRetrieveUserProfile() async throws {
        // Este test solo funcionará en dispositivos reales con biometría
        let userID = "test_user_001"
        let userName = "Fernando Test"
        let embeddings: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        let encryptedEmbedding = try embeddingManager.encrypt(embedding: embeddings)
        let metadata = UserProfileMetadata(
            userID: userID,
            name: userName,
            embeddingCount: 1
        )
        
        try await keychainManager.storeUserProfile(
            userID: userID,
            encryptedEmbeddings: [encryptedEmbedding],
            metadata: metadata
        )
        
        let retrievedEmbeddings = try await keychainManager.retrieveUserEmbeddings(userID: userID)
        let retrievedMetadata = try await keychainManager.retrieveUserMetadata(userID: userID)
        
        XCTAssertEqual(retrievedEmbeddings.count, 1)
        XCTAssertEqual(retrievedMetadata.userID, userID)
        XCTAssertEqual(retrievedMetadata.name, userName)
        
        let decryptedEmbedding = try embeddingManager.decrypt(encryptedEmbedding: retrievedEmbeddings[0])
        XCTAssertEqual(decryptedEmbedding, embeddings)
    }
    */
}
