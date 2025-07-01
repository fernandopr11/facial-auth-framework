// FacialAuthManager.swift
// API Principal del Framework de Autenticación Facial

import Foundation
import UIKit
import AVFoundation
import CoreImage

/// Manager principal del framework de autenticación facial
/// Orquesta todos los componentes y provee una API simple para desarrolladores
@available(iOS 14.0, *)
public final class FacialAuthManager {
    
    // MARK: - Types
    
    /// Estado del sistema de autenticación
    public enum AuthenticationState: Equatable {
        case notConfigured           // Sin configurar
        case configuring            // Configurando componentes
        case ready                  // Listo para autenticar
        case scanning               // Escaneando rostro
        case processing             // Procesando autenticación
        case userRegistration       // Registrando nuevo usuario
        case completed(UserProfile) // Autenticación exitosa
        case failed(AuthenticationError) // Falló la autenticación
        case cancelled              // Cancelado por usuario
        
        public static func == (lhs: AuthenticationState, rhs: AuthenticationState) -> Bool {
            switch (lhs, rhs) {
            case (.notConfigured, .notConfigured),
                 (.configuring, .configuring),
                 (.ready, .ready),
                 (.scanning, .scanning),
                 (.processing, .processing),
                 (.userRegistration, .userRegistration),
                 (.cancelled, .cancelled):
                return true
            case (.completed(let user1), .completed(let user2)):
                return user1.id == user2.id
            case (.failed, .failed):
                return true // Simplificado
            default:
                return false
            }
        }
    }
    
    /// Errores específicos de autenticación
    public enum AuthenticationError: Error, LocalizedError {
        case cameraNotAvailable
        case permissionDenied
        case noFaceDetected
        case multipleFacesDetected
        case faceQualityTooLow
        case noUserRegistered
        case userNotRecognized
        case systemError(String)
        case timeout
        case cancelled
        
        public var errorDescription: String? {
            switch self {
            case .cameraNotAvailable:
                return "Camera not available on this device"
            case .permissionDenied:
                return "Camera permission is required for facial authentication"
            case .noFaceDetected:
                return "No face detected. Please position your face in front of the camera"
            case .multipleFacesDetected:
                return "Multiple faces detected. Please ensure only one person is visible"
            case .faceQualityTooLow:
                return "Face quality too low. Please improve lighting and position"
            case .noUserRegistered:
                return "No users registered. Please register a user first"
            case .userNotRecognized:
                return "User not recognized. Please try again or register"
            case .systemError(let details):
                return "System error: \(details)"
            case .timeout:
                return "Authentication timed out. Please try again"
            case .cancelled:
                return "Authentication was cancelled"
            }
        }
    }
    
    /// Modo de operación del sistema
    public enum OperationMode {
        case authentication    // Solo autenticar usuarios existentes
        case userRegistration  // Solo registrar nuevos usuarios
        case both             // Autenticar o registrar automáticamente
    }
    
    /// Resultado de autenticación completo
    public struct AuthenticationResult {
        public let user: UserProfile
        public let confidence: Float
        public let processingTime: TimeInterval
        public let method: String // "face_recognition"
        public let timestamp: Date
        
        internal init(user: UserProfile, confidence: Float, processingTime: TimeInterval, method: String = "face_recognition") {
            self.user = user
            self.confidence = confidence
            self.processingTime = processingTime
            self.method = method
            self.timestamp = Date()
        }
    }
    
    /// Progreso de registro de usuario
    public struct RegistrationProgress {
        public let currentStep: Int
        public let totalSteps: Int
        public let currentStepDescription: String
        public let overallProgress: Float // 0.0 - 1.0
        
        internal init(currentStep: Int, totalSteps: Int, description: String) {
            self.currentStep = currentStep
            self.totalSteps = totalSteps
            self.currentStepDescription = description
            self.overallProgress = Float(currentStep) / Float(totalSteps)
        }
    }
    
    // MARK: - Delegate Protocol
    
    public protocol FacialAuthDelegate: AnyObject {
        /// Estado del sistema cambió
        func authManager(_ manager: FacialAuthManager, didChangeState state: AuthenticationState)
        
        /// Autenticación completada exitosamente
        func authManager(_ manager: FacialAuthManager, didAuthenticate result: AuthenticationResult)
        
        /// Falló la autenticación
        func authManager(_ manager: FacialAuthManager, didFailAuthentication error: AuthenticationError)
        
        /// Progreso de registro de usuario
        func authManager(_ manager: FacialAuthManager, registrationProgress: RegistrationProgress)
        
        /// Feedback visual para el usuario
        func authManager(_ manager: FacialAuthManager, didReceiveFeedback message: String, type: FeedbackType)
        
        /// Preview de cámara disponible
        func authManager(_ manager: FacialAuthManager, didUpdatePreview previewLayer: AVCaptureVideoPreviewLayer)
    }
    
    public enum FeedbackType {
        case guidance    // Guía para el usuario
        case warning     // Advertencia
        case success     // Éxito
        case error       // Error
    }
    
    // MARK: - Public Properties
    
    public weak var delegate: FacialAuthDelegate?
    public private(set) var currentState: AuthenticationState = .notConfigured
    public let configuration: FacialAuthConfiguration
    
    // MARK: - Private Properties
    
    // Core components
    private var cameraManager: TrueDepthCameraManager?
    private var realTimeProcessor: RealTimeProcessor?
    private var faceDetectionManager: FaceDetectionManager?
    private var embeddingExtractor: FaceEmbeddingExtractor?
    private var embeddingComparator: EmbeddingComparator?
    private var keychainManager: MultiUserKeychainManager?
    private var secureEmbeddingManager: SecureEmbeddingManager?
    
    // State management
    private var currentMode: OperationMode = .authentication
    private var authenticationStartTime: Date?
    private var registrationData: RegistrationData?
    
    // Processing queues
    private let mainQueue = DispatchQueue.main
    private let processingQueue = DispatchQueue(label: "facial.auth.processing", qos: .userInitiated)
    
    // Session management
    private var isSessionActive: Bool = false
    private var sessionTimeout: Timer?
    
    // MARK: - Private Types
    
    private struct RegistrationData {
        var userName: String
        var collectedEmbeddings: [FaceEmbeddingExtractor.ExtractionResult] = []
        var requiredEmbeddings: Int = 5
        var currentStep: Int = 0
        
        var isComplete: Bool {
            return collectedEmbeddings.count >= requiredEmbeddings
        }
        
        var progress: Float {
            return Float(collectedEmbeddings.count) / Float(requiredEmbeddings)
        }
    }
    
    // MARK: - Initialization
    
    /// Inicializa el manager con configuración específica
    /// - Parameter configuration: Configuración del framework
    public init(configuration: FacialAuthConfiguration = .default) {
        self.configuration = configuration
        setupComponents()
    }
    
    deinit {
        stopSession()
    }
    
    // MARK: - Public API
    
    /// Configura el sistema de autenticación
    public func configure() async throws {
        setState(.configuring)
        
        do {
            try await setupCoreComponents()
            setState(.ready)
        } catch {
            setState(.failed(.systemError(error.localizedDescription)))
            throw error
        }
    }
    
    /// Inicia autenticación de usuario existente
    /// - Parameter timeout: Tiempo límite en segundos (default: 30)
    public func startAuthentication(timeout: TimeInterval = 30.0) async throws {
        guard currentState == .ready else {
            throw AuthenticationError.systemError("System not ready")
        }
        
        // Verificar que hay usuarios registrados
        let users = try await keychainManager?.listAllUsers() ?? []
        guard !users.isEmpty else {
            throw AuthenticationError.noUserRegistered
        }
        
        currentMode = .authentication
        try await startSession(timeout: timeout)
    }
    
    /// Inicia registro de nuevo usuario
    /// - Parameters:
    ///   - userName: Nombre del usuario a registrar
    ///   - timeout: Tiempo límite en segundos (default: 60)
    public func startRegistration(userName: String, timeout: TimeInterval = 60.0) async throws {
        guard currentState == .ready else {
            throw AuthenticationError.systemError("System not ready")
        }
        
        guard !userName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AuthenticationError.systemError("User name cannot be empty")
        }
        
        // Verificar que el usuario no existe
        if let keychain = keychainManager, await keychain.userExists(userID: userName) {
            throw AuthenticationError.systemError("User already exists")
        }
        
        currentMode = .userRegistration
        registrationData = RegistrationData(userName: userName)
        
        try await startSession(timeout: timeout)
    }
    
    /// Inicia modo automático (autentica o registra según sea necesario)
    /// - Parameter timeout: Tiempo límite en segundos (default: 45)
    public func startSmartAuthentication(timeout: TimeInterval = 45.0) async throws {
        guard currentState == .ready else {
            throw AuthenticationError.systemError("System not ready")
        }
        
        currentMode = .both
        try await startSession(timeout: timeout)
    }
    
    /// Detiene la sesión actual
    public func stopSession() {
        sessionTimeout?.invalidate()
        sessionTimeout = nil
        
        cameraManager?.stopCapture()
        isSessionActive = false
        registrationData = nil
        
        if currentState != .ready && currentState != .notConfigured {
            setState(.ready)
        }
    }
    
    /// Cancela la operación actual
    public func cancel() {
        stopSession()
        setState(.cancelled)
    }
    
    /// Obtiene preview layer para mostrar en UI
    /// - Returns: Preview layer de la cámara
    public func getPreviewLayer() -> AVCaptureVideoPreviewLayer? {
        return cameraManager?.createPreviewLayer()
    }
    
    /// Lista todos los usuarios registrados
    /// - Returns: Array de nombres de usuario
    public func getRegisteredUsers() async throws -> [String] {
        guard let keychain = keychainManager else {
            throw AuthenticationError.systemError("Keychain not initialized")
        }
        
        return try await keychain.listAllUsers()
    }
    
    /// Elimina un usuario registrado
    /// - Parameter userName: Nombre del usuario a eliminar
    public func deleteUser(userName: String) async throws {
        guard let keychain = keychainManager else {
            throw AuthenticationError.systemError("Keychain not initialized")
        }
        
        try await keychain.deleteUser(userID: userName)
    }
    
    /// Verifica si el sistema está listo
    public var isReady: Bool {
        return currentState == .ready
    }
    
    /// Verifica si una sesión está activa
    public var isActive: Bool {
        return isSessionActive
    }
    
    // MARK: - Private Methods
    
    private func setupComponents() {
        // Los componentes se inicializarán en configure()
    }
    
    private func setupCoreComponents() async throws {
        // Inicializar managers de seguridad
        secureEmbeddingManager = try SecureEmbeddingManager()
        keychainManager = MultiUserKeychainManager()
        
        // Inicializar extractor de embeddings
        embeddingExtractor = try FaceEmbeddingExtractor()
        
        // Inicializar comparador
        embeddingComparator = EmbeddingComparator()
        
        // Configurar cámara
        let cameraConfig = TrueDepthCameraManager.CameraConfiguration.default
        cameraManager = TrueDepthCameraManager(configuration: cameraConfig)
        cameraManager?.delegate = self
        
        try await cameraManager?.configure()
        
        // Configurar procesador en tiempo real
        let processorConfig = RealTimeProcessor.Configuration.default
        realTimeProcessor = RealTimeProcessor(configuration: processorConfig)
        realTimeProcessor?.delegate = self
        
        // Configurar detector facial
        let faceDetectionConfig = FaceDetectionManager.Configuration.default
        faceDetectionManager = FaceDetectionManager(configuration: faceDetectionConfig)
        faceDetectionManager?.delegate = self
        
        // Notificar preview disponible
        if let previewLayer = cameraManager?.createPreviewLayer() {
            mainQueue.async {
                self.delegate?.authManager(self, didUpdatePreview: previewLayer)
            }
        }
    }
    
    private func startSession(timeout: TimeInterval) async throws {
        guard !isSessionActive else { return }
        
        isSessionActive = true
        authenticationStartTime = Date()
        
        // Configurar timeout
        sessionTimeout = Timer.scheduledTimer(withTimeInterval: timeout, repeats: false) { _ in
            Task {
                await self.handleTimeout()
            }
        }
        
        // Iniciar captura de cámara
        try cameraManager?.startCapture()
        
        setState(.scanning)
        
        // Enviar feedback inicial
        let feedbackMessage = currentMode == .userRegistration
            ? "Posiciona tu rostro en el centro para registro"
            : "Posiciona tu rostro en el centro para autenticación"
        
        mainQueue.async {
            self.delegate?.authManager(self, didReceiveFeedback: feedbackMessage, type: .guidance)
        }
    }
    
    private func setState(_ newState: AuthenticationState) {
        currentState = newState
        mainQueue.async {
            self.delegate?.authManager(self, didChangeState: newState)
        }
    }
    
    private func handleTimeout() async {
        stopSession()
        setState(.failed(.timeout))
        
        mainQueue.async {
            self.delegate?.authManager(self, didFailAuthentication: .timeout)
        }
    }
    
    private func processAuthenticationFrame(_ processedFrame: RealTimeProcessor.ProcessedFrame) async {
        guard isSessionActive else { return }
        
        do {
            setState(.processing)
            
            // Extraer embedding del rostro
            guard let extractionResult = try await extractEmbedding(from: processedFrame) else {
                setState(.scanning)
                return
            }
            
            switch currentMode {
            case .authentication:
                try await performAuthentication(with: extractionResult)
                
            case .userRegistration:
                try await performRegistration(with: extractionResult)
                
            case .both:
                try await performSmartAuthentication(with: extractionResult)
            }
            
        } catch {
            setState(.failed(.systemError(error.localizedDescription)))
            
            mainQueue.async {
                self.delegate?.authManager(self, didFailAuthentication: .systemError(error.localizedDescription))
            }
        }
    }
    
    private func extractEmbedding(from frame: RealTimeProcessor.ProcessedFrame) async throws -> FaceEmbeddingExtractor.ExtractionResult? {
        guard let extractor = embeddingExtractor else {
            throw AuthenticationError.systemError("Embedding extractor not initialized")
        }
        
        guard frame.quality.isAcceptable else {
            return nil // Frame no es de suficiente calidad
        }
        
        return try await extractor.extractEmbedding(from: frame.processedImage)
    }
    
    private func performAuthentication(with extractionResult: FaceEmbeddingExtractor.ExtractionResult) async throws {
        guard let keychain = keychainManager,
              let comparator = embeddingComparator,
              let secureManager = secureEmbeddingManager else {
            throw AuthenticationError.systemError("Components not initialized")
        }
        
        // Obtener todos los usuarios registrados
        let userIDs = try await keychain.listAllUsers()
        
        guard !userIDs.isEmpty else {
            throw AuthenticationError.noUserRegistered
        }
        
        // Preparar embeddings para comparación
        var userEmbeddings: [String: [[Float]]] = [:]
        
        for userID in userIDs {
            let encryptedEmbeddings = try await keychain.retrieveUserEmbeddings(userID: userID)
            let decryptedEmbeddings = try encryptedEmbeddings.map {
                try secureManager.decrypt(encryptedEmbedding: $0)
            }
            userEmbeddings[userID] = decryptedEmbeddings
        }
        
        // Realizar identificación 1:N
        let identificationResult = try await comparator.identify(
            queryEmbedding: extractionResult.embedding,
            against: userEmbeddings
        )
        
        if let userID = identificationResult.userID,
           let comparison = identificationResult.comparisonResult {
            
            // Usuario identificado
            let user = UserProfile(name: userID)
            let processingTime = Date().timeIntervalSince(authenticationStartTime ?? Date())
            
            let result = AuthenticationResult(
                user: user,
                confidence: comparison.confidence,
                processingTime: processingTime
            )
            
            stopSession()
            setState(.completed(user))
            
            mainQueue.async {
                self.delegate?.authManager(self, didAuthenticate: result)
            }
            
        } else {
            // Usuario no reconocido
            throw AuthenticationError.userNotRecognized
        }
    }
    
    private func performRegistration(with extractionResult: FaceEmbeddingExtractor.ExtractionResult) async throws {
        guard var regData = registrationData else {
            throw AuthenticationError.systemError("Registration data not found")
        }
        
        // Añadir embedding al conjunto de registro
        regData.collectedEmbeddings.append(extractionResult)
        regData.currentStep += 1
        registrationData = regData
        
        // Notificar progreso
        let progress = RegistrationProgress(
            currentStep: regData.currentStep,
            totalSteps: regData.requiredEmbeddings,
            description: "Capturando muestra \(regData.currentStep) de \(regData.requiredEmbeddings)"
        )
        
        mainQueue.async {
            self.delegate?.authManager(self, registrationProgress: progress)
        }
        
        // Verificar si está completo
        if regData.isComplete {
            try await completeRegistration(with: regData)
        } else {
            setState(.scanning)
            
            mainQueue.async {
                self.delegate?.authManager(self, didReceiveFeedback: "Mueve ligeramente tu cabeza y mantén la posición", type: .guidance)
            }
        }
    }
    
    private func completeRegistration(with regData: RegistrationData) async throws {
        guard let keychain = keychainManager,
              let secureManager = secureEmbeddingManager else {
            throw AuthenticationError.systemError("Components not initialized")
        }
        
        // Encriptar embeddings
        let encryptedEmbeddings = try regData.collectedEmbeddings.map { result in
            try secureManager.encrypt(embedding: result.embedding)
        }
        
        // Crear metadatos
        let metadata = UserProfileMetadata(
            userID: regData.userName,
            name: regData.userName,
            embeddingCount: encryptedEmbeddings.count
        )
        
        // Guardar en keychain
        try await keychain.storeUserProfile(
            userID: regData.userName,
            encryptedEmbeddings: encryptedEmbeddings,
            metadata: metadata
        )
        
        // Crear perfil de usuario
        let user = UserProfile(name: regData.userName)
        let processingTime = Date().timeIntervalSince(authenticationStartTime ?? Date())
        
        let result = AuthenticationResult(
            user: user,
            confidence: 1.0,
            processingTime: processingTime,
            method: "registration"
        )
        
        stopSession()
        setState(.completed(user))
        
        mainQueue.async {
            self.delegate?.authManager(self, didAuthenticate: result)
        }
    }
    
    private func performSmartAuthentication(with extractionResult: FaceEmbeddingExtractor.ExtractionResult) async throws {
        // Primero intentar autenticación
        do {
            try await performAuthentication(with: extractionResult)
        } catch AuthenticationError.noUserRegistered, AuthenticationError.userNotRecognized {
            // Si no hay usuarios o no se reconoce, cambiar a modo registro
            currentMode = .userRegistration
            registrationData = RegistrationData(userName: "User_\(Date().timeIntervalSince1970)")
            
            mainQueue.async {
                self.delegate?.authManager(self, didReceiveFeedback: "Usuario no registrado. Iniciando registro automático...", type: .guidance)
            }
            
            setState(.userRegistration)
            try await performRegistration(with: extractionResult)
        }
    }
}

// MARK: - TrueDepthCameraDelegate

extension FacialAuthManager: TrueDepthCameraManager.TrueDepthCameraDelegate {
    
    internal func cameraManager(_ manager: TrueDepthCameraManager, didCapture frame: TrueDepthCameraManager.CapturedFrame) {
        guard isSessionActive else { return }
        
        // Pasar frame al procesador en tiempo real
        realTimeProcessor?.processFrame(frame)
    }
    
    internal func cameraManager(_ manager: TrueDepthCameraManager, didChangeState state: TrueDepthCameraManager.CameraState) {
        switch state {
        case .failed(let error):
            mainQueue.async {
                self.delegate?.authManager(self, didFailAuthentication: .systemError(error.localizedDescription))
            }
        case .running:
            // Cámara lista
            break
        default:
            break
        }
    }
    
    internal func cameraManager(_ manager: TrueDepthCameraManager, didEncounterError error: Error) {
        mainQueue.async {
            self.delegate?.authManager(self, didFailAuthentication: .systemError(error.localizedDescription))
        }
    }
    
    internal func cameraManager(_ manager: TrueDepthCameraManager, didUpdateExposure info: TrueDepthCameraManager.ExposureInfo) {
        // Información de exposición - podría usarse para ajustes automáticos
    }
}

// MARK: - RealTimeProcessorDelegate

extension FacialAuthManager: RealTimeProcessor.RealTimeProcessorDelegate {
    
    internal func processor(_ processor: RealTimeProcessor, didProcess frame: RealTimeProcessor.ProcessedFrame) {
        guard isSessionActive else { return }
        
        // Pasar frame al detector facial
        Task {
            try await faceDetectionManager?.processImage(frame.processedImage, imageSize: frame.originalFrame.rgbImage.extent.size)
        }
        
        // Si el frame es de calidad suficiente, procesarlo para autenticación
        if frame.quality.isAcceptable && !frame.faceDetections.isEmpty {
            Task {
                await processAuthenticationFrame(frame)
            }
        }
    }
    
    internal func processor(_ processor: RealTimeProcessor, didChangeState state: RealTimeProcessor.ProcessingState) {
        // Manejar cambios de estado del procesador si es necesario
    }
    
    internal func processor(_ processor: RealTimeProcessor, didGenerateFeedback feedback: RealTimeProcessor.VisualFeedback) {
        let feedbackType: FeedbackType
        switch feedback.type {
        case .guidance:
            feedbackType = .guidance
        case .warning:
            feedbackType = .warning
        case .success:
            feedbackType = .success
        case .error:
            feedbackType = .error
        }
        
        mainQueue.async {
            self.delegate?.authManager(self, didReceiveFeedback: feedback.message, type: feedbackType)
        }
    }
    
    internal func processor(_ processor: RealTimeProcessor, didEncounterError error: Error) {
        mainQueue.async {
            self.delegate?.authManager(self, didFailAuthentication: .systemError(error.localizedDescription))
        }
    }
}

// MARK: - FaceDetectionDelegate

extension FacialAuthManager: FaceDetectionManager.FaceDetectionDelegate {
    
    internal func faceDetector(_ detector: FaceDetectionManager, didUpdateFace face: FaceDetectionManager.TrackedFace) {
        // Rostro actualizado - podría usarse para métricas
    }
    
    internal func faceDetector(_ detector: FaceDetectionManager, didDetectNewFace face: FaceDetectionManager.TrackedFace) {
        // Nuevo rostro detectado
        if !face.quality.isGoodForAuth {
            mainQueue.async {
                self.delegate?.authManager(self, didReceiveFeedback: "Ajusta tu posición para mejor calidad", type: .guidance)
            }
        }
    }
    
    internal func faceDetector(_ detector: FaceDetectionManager, didLoseFace faceID: UUID) {
        // Rostro perdido
        if isSessionActive && currentState == .scanning {
            mainQueue.async {
                self.delegate?.authManager(self, didReceiveFeedback: "Mantén tu rostro visible", type: .warning)
            }
        }
    }
    
    internal func faceDetector(_ detector: FaceDetectionManager, didGenerateGuide guide: FaceDetectionManager.VisualGuide) {
        let feedbackType: FeedbackType
        switch guide.type {
        case .perfect:
            feedbackType = .success
        case .moreLight:
            feedbackType = .warning
        case .openEyes:
            feedbackType = .warning
        default:
            feedbackType = .guidance
        }
        
        mainQueue.async {
            self.delegate?.authManager(self, didReceiveFeedback: guide.message, type: feedbackType)
        }
    }
    
    internal func faceDetector(_ detector: FaceDetectionManager, didEncounterError error: Error) {
        mainQueue.async {
            self.delegate?.authManager(self, didFailAuthentication: .systemError(error.localizedDescription))
        }
    }
}
