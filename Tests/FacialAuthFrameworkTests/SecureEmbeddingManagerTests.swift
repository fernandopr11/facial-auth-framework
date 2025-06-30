import XCTest
@testable import FacialAuthFramework

final class SecureEmbeddingManagerTests: XCTestCase {
    
    var secureManager: SecureEmbeddingManager!
    
    override func setUpWithError() throws {
        super.setUp()
        secureManager = try SecureEmbeddingManager()
    }
    
    override func tearDownWithError() throws {
        secureManager = nil
        super.tearDown()
    }
    
    func testEmbeddingEncryptionDecryption() throws {
        // Given: Un embedding facial simulado
        let originalEmbedding: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, 0.8, 0.9, 1.0]
        
        // When: Encriptamos el embedding
        let encryptedEmbedding = try secureManager.encrypt(embedding: originalEmbedding)
        
        // Then: Los datos encriptados deben ser diferentes
        XCTAssertFalse(encryptedEmbedding.encryptedData.isEmpty)
        XCTAssertEqual(encryptedEmbedding.salt.count, 32)
        XCTAssertEqual(encryptedEmbedding.tag.count, 16)
        
        // When: Desencriptamos
        let decryptedEmbedding = try secureManager.decrypt(encryptedEmbedding: encryptedEmbedding)
        
        // Then: Debe ser igual al original
        XCTAssertEqual(decryptedEmbedding.count, originalEmbedding.count)
        for (original, decrypted) in zip(originalEmbedding, decryptedEmbedding) {
            XCTAssertEqual(original, decrypted, accuracy: 0.0001)
        }
    }
    
    func testEmbeddingIntegrityVerification() throws {
        // Given: Un embedding válido encriptado
        let embedding: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let encryptedEmbedding = try secureManager.encrypt(embedding: embedding)
        
        // When: Verificamos integridad
        let isValid = secureManager.verifyIntegrity(of: encryptedEmbedding)
        
        // Then: Debe ser válido
        XCTAssertTrue(isValid)
    }
    
    func testEncryptedEmbeddingSerialization() throws {
        // Given: Un embedding encriptado
        let embedding: [Float] = [0.5, -0.3, 0.8, 1.2, -0.9]
        let encryptedEmbedding = try secureManager.encrypt(embedding: embedding)
        
        // When: Serializamos y deserializamos
        let serializedData = encryptedEmbedding.serialized
        let deserializedEmbedding = SecureEmbeddingManager.EncryptedEmbedding.from(serializedData)
        
        // Then: Debe poder desencriptarse correctamente
        XCTAssertNotNil(deserializedEmbedding)
        
        let decryptedEmbedding = try secureManager.decrypt(encryptedEmbedding: deserializedEmbedding!)
        
        XCTAssertEqual(decryptedEmbedding.count, embedding.count)
        for (original, decrypted) in zip(embedding, decryptedEmbedding) {
            XCTAssertEqual(original, decrypted, accuracy: 0.0001)
        }
    }
    
    func testDifferentEncryptionsProduceDifferentResults() throws {
        // Given: El mismo embedding
        let embedding: [Float] = [1.0, 2.0, 3.0]
        
        // When: Encriptamos dos veces
        let encrypted1 = try secureManager.encrypt(embedding: embedding)
        let encrypted2 = try secureManager.encrypt(embedding: embedding)
        
        // Then: Los resultados deben ser diferentes (debido a salt y nonce únicos)
        XCTAssertNotEqual(encrypted1.encryptedData, encrypted2.encryptedData)
        XCTAssertNotEqual(encrypted1.salt, encrypted2.salt)
        XCTAssertNotEqual(encrypted1.nonceData, encrypted2.nonceData)
        
        // But: Ambos deben desencriptarse al mismo resultado
        let decrypted1 = try secureManager.decrypt(encryptedEmbedding: encrypted1)
        let decrypted2 = try secureManager.decrypt(encryptedEmbedding: encrypted2)
        
        XCTAssertEqual(decrypted1, decrypted2)
    }
    
    func testEmptyEmbeddingEncryption() throws {
        // Given: Un embedding vacío
        let emptyEmbedding: [Float] = []
        
        // When & Then: Debe poder encriptar y desencriptar sin errores
        let encrypted = try secureManager.encrypt(embedding: emptyEmbedding)
        let decrypted = try secureManager.decrypt(encryptedEmbedding: encrypted)
        
        XCTAssertEqual(decrypted, emptyEmbedding)
        XCTAssertTrue(decrypted.isEmpty)
    }
    
    func testLargeEmbeddingEncryption() throws {
        // Given: Un embedding grande (simulando InceptionResNet embeddings de 512 dimensiones)
        let largeEmbedding: [Float] = (0..<512).map { Float($0) / 100.0 }
        
        // When: Encriptamos y desencriptamos
        let encrypted = try secureManager.encrypt(embedding: largeEmbedding)
        let decrypted = try secureManager.decrypt(encryptedEmbedding: encrypted)
        
        // Then: Debe mantener la precisión
        XCTAssertEqual(decrypted.count, 512)
        for (original, decrypted) in zip(largeEmbedding, decrypted) {
            XCTAssertEqual(original, decrypted, accuracy: 0.0001)
        }
    }
}
