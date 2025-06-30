// SecureEmbeddingManager.swift
// Gestión segura de embeddings faciales con encriptación AES-256

import Foundation
import CryptoKit
import UIKit

/// Gestor de encriptación para embeddings faciales
/// Implementa AES-256-GCM con llaves derivadas usando PBKDF2
@available(iOS 14.0, *)
internal final class SecureEmbeddingManager {
    
    // MARK: - Types
    
    /// Estructura para datos encriptados
    internal struct EncryptedEmbedding: Codable {
        let encryptedData: Data
        let nonceData: Data
        let salt: Data
        let tag: Data
        
        init(encryptedData: Data, nonce: AES.GCM.Nonce, salt: Data, tag: Data) {
            self.encryptedData = encryptedData
            self.nonceData = Data(nonce)
            self.salt = salt
            self.tag = tag
        }
        
        var nonce: AES.GCM.Nonce {
            try! AES.GCM.Nonce(data: nonceData)
        }
        
        /// Convierte a Data para almacenamiento
        var serialized: Data {
            try! JSONEncoder().encode(self)
        }
        
        /// Crea desde datos serializados
        static func from(_ data: Data) -> EncryptedEmbedding? {
            try? JSONDecoder().decode(EncryptedEmbedding.self, from: data)
        }
    }
    
    // MARK: - Constants
    
    private static let keyDerivationIterations: Int = 100_000
    private static let saltSize: Int = 32
    private static let keySize: Int = 32 // AES-256
    
    // MARK: - Private Properties
    
    private let masterKey: SymmetricKey
    
    // MARK: - Initialization
    
    /// Inicializa con clave maestra derivada de datos biométricos del dispositivo
    internal init() throws {
        // Generar clave maestra usando identificador único del dispositivo
        let deviceID = Self.getDeviceIdentifier()
        self.masterKey = try Self.deriveMasterKey(from: deviceID)
    }
    
    // MARK: - Public Methods
    
    /// Encripta un embedding facial
    /// - Parameter embedding: Array de Float con las características faciales
    /// - Returns: Datos encriptados para almacenamiento seguro
    internal func encrypt(embedding: [Float]) throws -> EncryptedEmbedding {
        // Convertir embedding a Data
        let embeddingData = embedding.withUnsafeBytes { Data($0) }
        
        // Generar salt único para esta operación
        let salt = Self.generateSalt()
        
        // Derivar clave específica usando salt
        let derivedKey = try Self.deriveKey(from: masterKey, salt: salt)
        
        // Generar nonce único
        let nonce = AES.GCM.Nonce()
        
        // Encriptar usando AES-GCM
        let sealedBox = try AES.GCM.seal(embeddingData, using: derivedKey, nonce: nonce)
        
        return EncryptedEmbedding(
            encryptedData: sealedBox.ciphertext,
            nonce: nonce,
            salt: salt,
            tag: sealedBox.tag
        )
    }
    
    /// Desencripta un embedding facial
    /// - Parameter encryptedEmbedding: Datos encriptados
    /// - Returns: Array de Float con las características faciales
    internal func decrypt(encryptedEmbedding: EncryptedEmbedding) throws -> [Float] {
        // Derivar la misma clave usando el salt almacenado
        let derivedKey = try Self.deriveKey(from: masterKey, salt: encryptedEmbedding.salt)
        
        // Recrear sealed box
        let sealedBox = try AES.GCM.SealedBox(
            nonce: encryptedEmbedding.nonce,
            ciphertext: encryptedEmbedding.encryptedData,
            tag: encryptedEmbedding.tag
        )
        
        // Desencriptar
        let decryptedData = try AES.GCM.open(sealedBox, using: derivedKey)
        
        // Convertir de vuelta a [Float]
        return decryptedData.withUnsafeBytes { bytes in
            let floatPointer = bytes.bindMemory(to: Float.self)
            return Array(floatPointer)
        }
    }
    
    /// Verifica la integridad de un embedding encriptado
    /// - Parameter encryptedEmbedding: Datos a verificar
    /// - Returns: true si la integridad está intacta
    internal func verifyIntegrity(of encryptedEmbedding: EncryptedEmbedding) -> Bool {
        do {
            _ = try decrypt(encryptedEmbedding: encryptedEmbedding)
            return true
        } catch {
            return false
        }
    }
    
    // MARK: - Private Methods
    
    /// Genera un salt criptográficamente seguro
    private static func generateSalt() -> Data {
        var salt = Data(count: saltSize)
        let result = salt.withUnsafeMutableBytes { bytes in
            SecRandomCopyBytes(kSecRandomDefault, saltSize, bytes.baseAddress!)
        }
        assert(result == errSecSuccess, "Failed to generate secure salt")
        return salt
    }
    
    /// Deriva una clave usando PBKDF2
    private static func deriveKey(from masterKey: SymmetricKey, salt: Data) throws -> SymmetricKey {
        let masterKeyData = masterKey.withUnsafeBytes { Data($0) }
        
        var derivedKey = Data(count: keySize)
        let result = derivedKey.withUnsafeMutableBytes { derivedKeyBytes in
            salt.withUnsafeBytes { saltBytes in
                masterKeyData.withUnsafeBytes { masterKeyBytes in
                    CCKeyDerivationPBKDF(
                        CCPBKDFAlgorithm(kCCPBKDF2),
                        masterKeyBytes.baseAddress, masterKeyData.count,
                        saltBytes.baseAddress, salt.count,
                        CCPseudoRandomAlgorithm(kCCPRFHmacAlgSHA256),
                        UInt32(keyDerivationIterations),
                        derivedKeyBytes.baseAddress, keySize
                    )
                }
            }
        }
        
        guard result == kCCSuccess else {
            throw FacialAuthError.encryptionFailed
        }
        
        return SymmetricKey(data: derivedKey)
    }
    
    /// Deriva la clave maestra del dispositivo
    private static func deriveMasterKey(from deviceID: String) throws -> SymmetricKey {
        guard let deviceData = deviceID.data(using: .utf8) else {
            throw FacialAuthError.encryptionFailed
        }
        
        // Usar SHA256 para crear clave consistente
        let hashedData = SHA256.hash(data: deviceData)
        return SymmetricKey(data: hashedData)
    }
    
    /// Obtiene identificador único del dispositivo
    private static func getDeviceIdentifier() -> String {
        #if canImport(UIKit)
        // En iOS, usar identifierForVendor
        return UIDevice.current.identifierForVendor?.uuidString ?? "default-device-id"
        #else
        // Fallback para macOS (desarrollo)
        return "development-device-id"
        #endif
    }
}

// MARK: - CommonCrypto Import
#if canImport(CommonCrypto)
import CommonCrypto
import UIKit
#endif
