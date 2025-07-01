// AuthConfiguration.swift
// Configuración completa del framework de autenticación facial

import Foundation
import AVFoundation

#if canImport(UIKit)
import UIKit
#endif

/// Configuración operacional del framework de autenticación facial
/// La seguridad siempre es máxima - solo se configuran aspectos operacionales
@available(iOS 14.0, *)
public struct FacialAuthConfiguration {
    
    // MARK: - Camera Configuration
    
    /// Calidad de la cámara
    public enum CameraQuality {
        case standard      // 640x480 - rápido, menor calidad
        case high         // 1280x720 - balanceado
        case maximum      // Máxima resolución disponible
        
        internal var sessionPreset: AVCaptureSession.Preset {
            switch self {
            case .standard:
                return .vga640x480
            case .high:
                return .hd1280x720
            case .maximum:
                return .hd1920x1080
            }
        }
        
        internal var frameRate: Int32 {
            switch self {
            case .standard:
                return 15  // Menor consumo de batería
            case .high:
                return 30  // Fluido
            case .maximum:
                return 30  // Máxima calidad
            }
        }
    }
    
    // MARK: - Performance Configuration
    
    /// Perfil de performance
    public enum PerformanceProfile {
        case battery      // Optimizado para batería
        case balanced     // Balanceado
        case speed        // Máxima velocidad
        
        internal var bufferSize: Int {
            switch self {
            case .battery: return 3
            case .balanced: return 5
            case .speed: return 10
            }
        }
        
        internal var processingInterval: Int {
            switch self {
            case .battery: return 5  // Procesar cada 5 frames
            case .balanced: return 3  // Procesar cada 3 frames
            case .speed: return 1     // Procesar cada frame
            }
        }
    }
    
    // MARK: - Timeout Configuration
    
    /// Configuración de timeouts
    public struct TimeoutConfiguration {
        public let authentication: TimeInterval    // Tiempo límite para autenticación
        public let registration: TimeInterval      // Tiempo límite para registro
        public let faceDetection: TimeInterval     // Tiempo límite para detectar rostro
        public let processing: TimeInterval        // Tiempo límite por frame
        
        public init(
            authentication: TimeInterval = 30.0,
            registration: TimeInterval = 60.0,
            faceDetection: TimeInterval = 10.0,
            processing: TimeInterval = 0.5
        ) {
            self.authentication = authentication
            self.registration = registration
            self.faceDetection = faceDetection
            self.processing = processing
        }
        
        /// Timeouts conservadores para conexiones lentas
        public static let relaxed = TimeoutConfiguration(
            authentication: 45.0,
            registration: 90.0,
            faceDetection: 15.0,
            processing: 1.0
        )
        
        /// Timeouts estrictos para aplicaciones críticas
        public static let strict = TimeoutConfiguration(
            authentication: 15.0,
            registration: 30.0,
            faceDetection: 5.0,
            processing: 0.2
        )
        
        /// Timeouts balanceados (default)
        public static let balanced = TimeoutConfiguration()
    }
    
    // MARK: - UI Configuration
    
    /// Configuración de interfaz
    public struct UIConfiguration {
        public let showPreview: Bool               // Mostrar preview de cámara
        public let showGuidance: Bool              // Mostrar guías visuales
        public let showProgress: Bool              // Mostrar progreso de registro
        public let allowManualCapture: Bool        // Permitir captura manual
        public let hapticFeedback: Bool            // Feedback háptico
        
        public init(
            showPreview: Bool = true,
            showGuidance: Bool = true,
            showProgress: Bool = true,
            allowManualCapture: Bool = false,
            hapticFeedback: Bool = true
        ) {
            self.showPreview = showPreview
            self.showGuidance = showGuidance
            self.showProgress = showProgress
            self.allowManualCapture = allowManualCapture
            self.hapticFeedback = hapticFeedback
        }
        
        /// UI completa para mejor experiencia
        public static let full = UIConfiguration()
        
        /// UI mínima para aplicaciones simples
        public static let minimal = UIConfiguration(
            showPreview: true,
            showGuidance: false,
            showProgress: false,
            allowManualCapture: false,
            hapticFeedback: false
        )
        
        /// UI solo para debugging
        public static let debug = UIConfiguration(
            showPreview: true,
            showGuidance: true,
            showProgress: true,
            allowManualCapture: true,
            hapticFeedback: false
        )
    }
    
    // MARK: - Authentication Configuration
    
    /// Configuración de autenticación
    public struct AuthenticationConfiguration {
        public let confidenceThreshold: Float     // Threshold de confianza (0.0 - 1.0)
        public let allowMultipleFaces: Bool        // Permitir múltiples rostros
        public let requiredStability: Int          // Frames requeridos de estabilidad
        public let maxRetries: Int                 // Intentos máximos antes de fallar
        
        public init(
            confidenceThreshold: Float = 0.85,
            allowMultipleFaces: Bool = false,
            requiredStability: Int = 5,
            maxRetries: Int = 3
        ) {
            self.confidenceThreshold = confidenceThreshold
            self.allowMultipleFaces = allowMultipleFaces
            self.requiredStability = requiredStability
            self.maxRetries = maxRetries
        }
        
        /// Configuración estricta para alta seguridad
        public static let strict = AuthenticationConfiguration(
            confidenceThreshold: 0.95,
            allowMultipleFaces: false,
            requiredStability: 10,
            maxRetries: 2
        )
        
        /// Configuración balanceada (default)
        public static let balanced = AuthenticationConfiguration()
        
        /// Configuración para testing/development
        public static let development = AuthenticationConfiguration(
            confidenceThreshold: 0.7,
            allowMultipleFaces: true,
            requiredStability: 3,
            maxRetries: 5
        )
    }
    
    // MARK: - Registration Configuration
    
    /// Configuración de registro de usuarios
    public struct RegistrationConfiguration {
        public let requiredSamples: Int            // Muestras requeridas por usuario
        public let sampleVariationRequired: Bool   // Requerir variación entre muestras
        public let autoSaveProgress: Bool          // Guardar progreso automáticamente
        public let allowOverwrite: Bool            // Permitir sobrescribir usuario existente
        
        public init(
            requiredSamples: Int = 5,
            sampleVariationRequired: Bool = true,
            autoSaveProgress: Bool = true,
            allowOverwrite: Bool = false
        ) {
            self.requiredSamples = requiredSamples
            self.sampleVariationRequired = sampleVariationRequired
            self.autoSaveProgress = autoSaveProgress
            self.allowOverwrite = allowOverwrite
        }
        
        /// Registro rápido con menos muestras
        public static let quick = RegistrationConfiguration(
            requiredSamples: 3,
            sampleVariationRequired: false,
            autoSaveProgress: true,
            allowOverwrite: false
        )
        
        /// Registro completo para máxima precisión
        public static let comprehensive = RegistrationConfiguration(
            requiredSamples: 8,
            sampleVariationRequired: true,
            autoSaveProgress: true,
            allowOverwrite: false
        )
        
        /// Configuración balanceada (default)
        public static let balanced = RegistrationConfiguration()
    }
    
    // MARK: - Main Configuration Properties
    
    public let cameraQuality: CameraQuality
    public let performance: PerformanceProfile
    public let timeouts: TimeoutConfiguration
    public let ui: UIConfiguration
    public let authentication: AuthenticationConfiguration
    public let registration: RegistrationConfiguration
    
    // Propiedades de compatibilidad con la versión anterior
    public var securityLevel: SecurityLevel { .high } // Siempre alta seguridad
    public var confidenceThreshold: Float { authentication.confidenceThreshold }
    public var allowMultipleFaces: Bool { authentication.allowMultipleFaces }
    public var cameraTimeout: TimeInterval { timeouts.authentication }
    
    /// Nivel de seguridad (siempre alto)
    public enum SecurityLevel: CaseIterable {
        case high // Solo alta seguridad disponible
        
        public static let `default`: SecurityLevel = .high
    }
    
    // MARK: - Initialization
    
    /// Inicializador completo con todas las opciones
    public init(
        cameraQuality: CameraQuality = .high,
        performance: PerformanceProfile = .balanced,
        timeouts: TimeoutConfiguration = .balanced,
        ui: UIConfiguration = .full,
        authentication: AuthenticationConfiguration = .balanced,
        registration: RegistrationConfiguration = .balanced
    ) {
        self.cameraQuality = cameraQuality
        self.performance = performance
        self.timeouts = timeouts
        self.ui = ui
        self.authentication = authentication
        self.registration = registration
        
        // Validar configuración
        validateConfiguration()
    }
    
    // MARK: - Preset Configurations
    
    /// Configuración por defecto - compatible con versión anterior
    public static let `default` = FacialAuthConfiguration(
        cameraQuality: .high,
        performance: .balanced,
        timeouts: .balanced,
        ui: .full,
        authentication: AuthenticationConfiguration(
            confidenceThreshold: 0.85,
            allowMultipleFaces: false,
            requiredStability: 5,
            maxRetries: 3
        ),
        registration: .balanced
    )
    
    /// Configuración optimizada para velocidad máxima
    public static let speed = FacialAuthConfiguration(
        cameraQuality: .standard,
        performance: .speed,
        timeouts: .strict,
        ui: .minimal,
        authentication: .balanced,
        registration: .quick
    )
    
    /// Configuración optimizada para calidad máxima
    public static let quality = FacialAuthConfiguration(
        cameraQuality: .maximum,
        performance: .balanced,
        timeouts: .relaxed,
        ui: .full,
        authentication: .strict,
        registration: .comprehensive
    )
    
    /// Configuración optimizada para batería
    public static let battery = FacialAuthConfiguration(
        cameraQuality: .standard,
        performance: .battery,
        timeouts: .relaxed,
        ui: .minimal,
        authentication: .balanced,
        registration: .quick
    )
    
    /// Configuración para desarrollo y testing
    public static let development = FacialAuthConfiguration(
        cameraQuality: .high,
        performance: .balanced,
        timeouts: .relaxed,
        ui: .debug,
        authentication: .development,
        registration: .balanced
    )
    
    /// Configuración para aplicaciones enterprise críticas
    public static let enterprise = FacialAuthConfiguration(
        cameraQuality: .maximum,
        performance: .balanced,
        timeouts: .strict,
        ui: .full,
        authentication: .strict,
        registration: .comprehensive
    )
    
    // MARK: - Validation
    
    /// Valida que la configuración sea consistente
    private func validateConfiguration() {
        // Validar confidence threshold
        assert(authentication.confidenceThreshold >= 0.0 && authentication.confidenceThreshold <= 1.0,
               "Confidence threshold must be between 0.0 and 1.0")
        
        // Validar timeouts
        assert(timeouts.authentication > 0, "Authentication timeout must be positive")
        assert(timeouts.registration > 0, "Registration timeout must be positive")
        assert(timeouts.faceDetection > 0, "Face detection timeout must be positive")
        assert(timeouts.processing > 0, "Processing timeout must be positive")
        
        // Validar registration samples
        assert(registration.requiredSamples > 0 && registration.requiredSamples <= 20,
               "Required samples must be between 1 and 20")
        
        // Validar stability frames
        assert(authentication.requiredStability > 0 && authentication.requiredStability <= 30,
               "Required stability frames must be between 1 and 30")
        
        // Validar max retries
        assert(authentication.maxRetries > 0 && authentication.maxRetries <= 10,
               "Max retries must be between 1 and 10")
    }
    
    // MARK: - Configuration Info
    
    /// Información descriptiva de la configuración actual
    public var description: String {
        return """
        FacialAuth Configuration:
        - Camera: \(cameraQuality)
        - Performance: \(performance)
        - Confidence: \(authentication.confidenceThreshold)
        - Samples: \(registration.requiredSamples)
        - Auth Timeout: \(timeouts.authentication)s
        - Registration Timeout: \(timeouts.registration)s
        """
    }
    
    /// Verifica si la configuración es compatible con el dispositivo
    public func isCompatibleWithDevice() -> Bool {
        // Verificar capacidades del dispositivo
        #if canImport(UIKit)
        let hasCamera = UIImagePickerController.isSourceTypeAvailable(.camera)
        #else
        let hasCamera = false
        #endif
        
        // Para configuraciones de máxima calidad, verificar TrueDepth
        if cameraQuality == .maximum {
            return hasCamera && isTrueDepthAvailable()
        }
        
        return hasCamera
    }
    
    private func isTrueDepthAvailable() -> Bool {
        #if canImport(AVFoundation)
        return AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) != nil
        #else
        return false
        #endif
    }
    
    /// Configuración optimizada para el dispositivo actual
    public static func optimizedForCurrentDevice() -> FacialAuthConfiguration {
        let hasCamera = {
            #if canImport(UIKit)
            return UIImagePickerController.isSourceTypeAvailable(.camera)
            #else
            return false
            #endif
        }()
        
        let hasTrueDepth = {
            #if canImport(AVFoundation)
            return AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) != nil
            #else
            return false
            #endif
        }()
        
        if hasCamera && hasTrueDepth {
            // Dispositivo con TrueDepth - usar calidad alta
            return .quality
        } else if hasCamera {
            // Dispositivo estándar - optimizar para velocidad
            return .speed
        } else {
            // Sin cámara - configuración básica
            return .default
        }
    }
}

// MARK: - Extensions

extension FacialAuthConfiguration.CameraQuality: CustomStringConvertible {
    public var description: String {
        switch self {
        case .standard: return "Standard (640x480@15fps)"
        case .high: return "High (1280x720@30fps)"
        case .maximum: return "Maximum (1920x1080@30fps)"
        }
    }
}

extension FacialAuthConfiguration.PerformanceProfile: CustomStringConvertible {
    public var description: String {
        switch self {
        case .battery: return "Battery Optimized"
        case .balanced: return "Balanced"
        case .speed: return "Speed Optimized"
        }
    }
}
