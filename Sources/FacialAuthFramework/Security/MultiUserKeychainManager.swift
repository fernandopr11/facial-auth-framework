// MultiUserKeychainManager.swift
// Gestión segura de múltiples perfiles de usuario en Keychain

import Foundation
import Security
import LocalAuthentication

// MARK: - Security Constants (En caso de que no estén disponibles)
#if !canImport(Security)
private let errSecUserCancel: OSStatus = -128
private let errSecBiometryNotAvailable: OSStatus = -25293
private let errSecBiometryNotEnrolled: OSStatus = -25295
private let errSecBiometryLockout: OSStatus = -25292
#endif

/// Gestor de Keychain para múltiples usuarios con seguridad biométrica
/// Implementa buenas prácticas de seguridad y namespacing por app
@available(iOS 14.0, *)
internal final class MultiUserKeychainManager {
    
    // MARK: - Types
    
    /// Política de acceso biométrico
    enum BiometricPolicy {
        case deviceOwnerAuthentication
        case deviceOwnerAuthenticationWithBiometrics
        case biometricAny
        case biometricCurrentSet
        
        var secAccessControl: SecAccessControl? {
            var flags: SecAccessControlCreateFlags
            
            switch self {
            case .deviceOwnerAuthentication:
                flags = .devicePasscode
            case .deviceOwnerAuthenticationWithBiometrics:
                flags = [.devicePasscode, .biometryAny]
            case .biometricAny:
                flags = .biometryAny
            case .biometricCurrentSet:
                flags = .biometryCurrentSet
            }
            
            var error: Unmanaged<CFError>?
            let accessControl = SecAccessControlCreateWithFlags(
                kCFAllocatorDefault,
                kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
                flags,
                &error
            )
            
            if let error = error?.takeRetainedValue() {
                print("⚠️ Error creating access control: \(error)")
                return nil
            }
            
            return accessControl
        }
    }
    
    /// Errores específicos del Keychain
    enum KeychainError: Error, LocalizedError {
        case itemNotFound
        case duplicateItem
        case invalidData
        case authenticationFailed
        case biometricNotAvailable
        case biometricNotEnrolled
        case biometricLockout
        case unexpectedError(OSStatus)
        
        var errorDescription: String? {
            switch self {
            case .itemNotFound:
                return "User profile not found in keychain"
            case .duplicateItem:
                return "User profile already exists"
            case .invalidData:
                return "Invalid data format"
            case .authenticationFailed:
                return "Biometric authentication failed"
            case .biometricNotAvailable:
                return "Biometric authentication not available"
            case .biometricNotEnrolled:
                return "No biometric data enrolled"
            case .biometricLockout:
                return "Biometric authentication locked out"
            case .unexpectedError(let status):
                return "Keychain error: \(status)"
            }
        }
    }
    
    // MARK: - Constants
    
    private static let servicePrefix = "com.facialauth.framework"
    private static let userProfilesKey = "user_profiles"
    private static let embeddingKeySuffix = "_embeddings"
    private static let metadataKeySuffix = "_metadata"
    
    // MARK: - Properties
    
    private let biometricPolicy: BiometricPolicy
    private let appIdentifier: String
    
    // MARK: - Initialization
    
    /// Inicializa el gestor con política biométrica específica
    /// - Parameters:
    ///   - biometricPolicy: Política de acceso biométrico
    ///   - appIdentifier: Identificador único de la app (para namespacing)
    internal init(
        biometricPolicy: BiometricPolicy = .biometricCurrentSet,
        appIdentifier: String = Bundle.main.bundleIdentifier ?? "default"
    ) {
        self.biometricPolicy = biometricPolicy
        self.appIdentifier = appIdentifier
    }
    
    // MARK: - User Profile Management
    
    /// Almacena embeddings de un usuario de forma segura
    /// - Parameters:
    ///   - userID: Identificador único del usuario
    ///   - encryptedEmbeddings: Embeddings encriptados del usuario
    ///   - metadata: Metadatos del perfil de usuario
    internal func storeUserProfile(
        userID: String,
        encryptedEmbeddings: [SecureEmbeddingManager.EncryptedEmbedding],
        metadata: UserProfileMetadata
    ) async throws {
        let embeddingKey = makeKey(for: userID, suffix: Self.embeddingKeySuffix)
        let metadataKey = makeKey(for: userID, suffix: Self.metadataKeySuffix)
        
        // Serializar embeddings
        let embeddingsData = try JSONEncoder().encode(encryptedEmbeddings)
        let metadataData = try JSONEncoder().encode(metadata)
        
        // Almacenar embeddings con protección biométrica
        try await storeSecureData(embeddingsData, forKey: embeddingKey)
        
        // Almacenar metadatos con protección biométrica
        try await storeSecureData(metadataData, forKey: metadataKey)
        
        // Actualizar lista de usuarios
        try await addUserToRegistry(userID: userID)
    }
    
    /// Recupera embeddings de un usuario
    /// - Parameter userID: Identificador del usuario
    /// - Returns: Array de embeddings encriptados
    internal func retrieveUserEmbeddings(userID: String) async throws -> [SecureEmbeddingManager.EncryptedEmbedding] {
        let embeddingKey = makeKey(for: userID, suffix: Self.embeddingKeySuffix)
        
        let data = try await retrieveSecureData(forKey: embeddingKey)
        
        return try JSONDecoder().decode([SecureEmbeddingManager.EncryptedEmbedding].self, from: data)
    }
    
    /// Recupera metadatos de un usuario
    /// - Parameter userID: Identificador del usuario
    /// - Returns: Metadatos del perfil
    internal func retrieveUserMetadata(userID: String) async throws -> UserProfileMetadata {
        let metadataKey = makeKey(for: userID, suffix: Self.metadataKeySuffix)
        
        let data = try await retrieveSecureData(forKey: metadataKey)
        
        return try JSONDecoder().decode(UserProfileMetadata.self, from: data)
    }
    
    /// Lista todos los usuarios registrados
    /// - Returns: Array de IDs de usuario
    internal func listAllUsers() async throws -> [String] {
        let registryKey = makeKey(for: Self.userProfilesKey, suffix: "")
        
        do {
            let data = try await retrieveSecureData(forKey: registryKey)
            return try JSONDecoder().decode([String].self, from: data)
        } catch KeychainError.itemNotFound {
            return []
        }
    }
    
    /// Elimina un usuario y todos sus datos
    /// - Parameter userID: Identificador del usuario a eliminar
    internal func deleteUser(userID: String) async throws {
        let embeddingKey = makeKey(for: userID, suffix: Self.embeddingKeySuffix)
        let metadataKey = makeKey(for: userID, suffix: Self.metadataKeySuffix)
        
        // Eliminar embeddings y metadatos
        try await deleteSecureData(forKey: embeddingKey)
        try await deleteSecureData(forKey: metadataKey)
        
        // Remover de la lista de usuarios
        try await removeUserFromRegistry(userID: userID)
    }
    
    /// Actualiza embeddings de un usuario existente
    /// - Parameters:
    ///   - userID: Identificador del usuario
    ///   - newEmbeddings: Nuevos embeddings encriptados
    internal func updateUserEmbeddings(
        userID: String,
        newEmbeddings: [SecureEmbeddingManager.EncryptedEmbedding]
    ) async throws {
        let embeddingKey = makeKey(for: userID, suffix: Self.embeddingKeySuffix)
        let embeddingsData = try JSONEncoder().encode(newEmbeddings)
        
        try await updateSecureData(embeddingsData, forKey: embeddingKey)
    }
    
    /// Verifica si un usuario existe
    /// - Parameter userID: Identificador del usuario
    /// - Returns: true si el usuario existe
    internal func userExists(userID: String) async -> Bool {
        let users = (try? await listAllUsers()) ?? []
        return users.contains(userID)
    }
    
    // MARK: - Cleanup and Maintenance
    
    /// Limpia datos temporales y optimiza almacenamiento
    internal func performMaintenance() async throws {
        // Verificar integridad de todos los usuarios
        let allUsers = try await listAllUsers()
        var validUsers: [String] = []
        
        for userID in allUsers {
            do {
                _ = try await retrieveUserMetadata(userID: userID)
                _ = try await retrieveUserEmbeddings(userID: userID)
                validUsers.append(userID)
            } catch {
                print("⚠️ Removing corrupted user data for: \(userID)")
                try? await deleteUser(userID: userID)
            }
        }
        
        // Actualizar registry con usuarios válidos
        if validUsers.count != allUsers.count {
            try await updateUserRegistry(validUsers)
        }
    }
    
    /// Elimina todos los datos de la app (para testing o reset)
    internal func deleteAllData() async throws {
        let allUsers = try await listAllUsers()
        
        for userID in allUsers {
            try await deleteUser(userID: userID)
        }
        
        // Eliminar registry
        let registryKey = makeKey(for: Self.userProfilesKey, suffix: "")
        try await deleteSecureData(forKey: registryKey)
    }
    
    // MARK: - Private Methods
    
    private func makeKey(for identifier: String, suffix: String) -> String {
        return "\(Self.servicePrefix).\(appIdentifier).\(identifier)\(suffix)"
    }
    
    private func storeSecureData(_ data: Data, forKey key: String) async throws {
        guard let accessControl = biometricPolicy.secAccessControl else {
            throw KeychainError.biometricNotAvailable
        }
        
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: key,
            kSecAttrAccessControl as String: accessControl,
            kSecValueData as String: data
        ]
        
        let status = SecItemAdd(query as CFDictionary, nil)
        
        switch status {
        case errSecSuccess:
            return
        case errSecDuplicateItem:
            // Si ya existe, actualizar
            try await updateSecureData(data, forKey: key)
        default:
            throw KeychainError.unexpectedError(status)
        }
    }
    
    private func retrieveSecureData(forKey key: String) async throws -> Data {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: key,
            kSecMatchLimit as String: kSecMatchLimitOne,
            kSecReturnData as String: true
        ]
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        switch status {
        case errSecSuccess:
            guard let data = result as? Data else {
                throw KeychainError.invalidData
            }
            return data
        case errSecItemNotFound:
            throw KeychainError.itemNotFound
        case -128: // errSecUserCancel
            throw KeychainError.authenticationFailed
        case -25293: // errSecBiometryNotAvailable
            throw KeychainError.biometricNotAvailable
        case -25295: // errSecBiometryNotEnrolled
            throw KeychainError.biometricNotEnrolled
        case -25292: // errSecBiometryLockout
            throw KeychainError.biometricLockout
        default:
            throw KeychainError.unexpectedError(status)
        }
    }
    
    private func updateSecureData(_ data: Data, forKey key: String) async throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: key
        ]
        
        let updateAttributes: [String: Any] = [
            kSecValueData as String: data
        ]
        
        let status = SecItemUpdate(query as CFDictionary, updateAttributes as CFDictionary)
        
        guard status == errSecSuccess else {
            throw KeychainError.unexpectedError(status)
        }
    }
    
    private func deleteSecureData(forKey key: String) async throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: key
        ]
        
        let status = SecItemDelete(query as CFDictionary)
        
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.unexpectedError(status)
        }
    }
    
    private func addUserToRegistry(userID: String) async throws {
        var users = (try? await listAllUsers()) ?? []
        
        if !users.contains(userID) {
            users.append(userID)
            try await updateUserRegistry(users)
        }
    }
    
    private func removeUserFromRegistry(userID: String) async throws {
        var users = try await listAllUsers()
        users.removeAll { $0 == userID }
        try await updateUserRegistry(users)
    }
    
    private func updateUserRegistry(_ users: [String]) async throws {
        let registryKey = makeKey(for: Self.userProfilesKey, suffix: "")
        let data = try JSONEncoder().encode(users)
        try await storeSecureData(data, forKey: registryKey)
    }
}

// MARK: - User Profile Metadata

/// Metadatos del perfil de usuario
internal struct UserProfileMetadata: Codable {
    let userID: String
    let name: String
    let dateCreated: Date
    let lastUpdated: Date
    let embeddingCount: Int
    let version: String
    
    internal init(userID: String, name: String, embeddingCount: Int) {
        self.userID = userID
        self.name = name
        self.dateCreated = Date()
        self.lastUpdated = Date()
        self.embeddingCount = embeddingCount
        self.version = FacialAuthFramework.version
    }
}
