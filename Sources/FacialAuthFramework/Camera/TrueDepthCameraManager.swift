// TrueDepthCameraManager.swift
// Gestor avanzado de cámara TrueDepth para captura facial segura

import Foundation
import AVFoundation
import CoreImage
import UIKit

#if canImport(UIKit)
import UIKit
#endif

/// Gestor especializado para cámara TrueDepth con captura RGB + Depth
/// Optimizado para autenticación facial en tiempo real
@available(iOS 14.0, *)
internal final class TrueDepthCameraManager: NSObject {
    
    // MARK: - Types
    
    /// Estado de la cámara
    internal enum CameraState: Equatable {
        case notConfigured
        case configuring
        case configured
        case running
        case stopped
        case failed(Error)
        
        // Implementar Equatable manualmente para el caso .failed
        static func == (lhs: CameraState, rhs: CameraState) -> Bool {
            switch (lhs, rhs) {
            case (.notConfigured, .notConfigured),
                 (.configuring, .configuring),
                 (.configured, .configured),
                 (.running, .running),
                 (.stopped, .stopped):
                return true
            case (.failed, .failed):
                return true // Simplificado: todos los errores se consideran iguales
            default:
                return false
            }
        }
    }
    /// Configuración de la cámara
    internal struct CameraConfiguration {
        let sessionPreset: AVCaptureSession.Preset
        let enableDepthCapture: Bool
        let enableFaceDetection: Bool
        let autoExposure: Bool
        let autoFocus: Bool
        let frameRate: Int32
        
        internal static let `default` = CameraConfiguration(
            sessionPreset: .vga640x480,
            enableDepthCapture: true,
            enableFaceDetection: true,
            autoExposure: true,
            autoFocus: true,
            frameRate: 30
        )
        
        internal static let highQuality = CameraConfiguration(
            sessionPreset: .hd1280x720,
            enableDepthCapture: true,
            enableFaceDetection: true,
            autoExposure: true,
            autoFocus: true,
            frameRate: 30
        )
        
        internal static let performanceOptimized = CameraConfiguration(
            sessionPreset: .vga640x480,
            enableDepthCapture: false,
            enableFaceDetection: true,
            autoExposure: false,
            autoFocus: false,
            frameRate: 15
        )
    }
    
    /// Frame capturado con datos RGB y depth
    internal struct CapturedFrame {
        let rgbImage: CIImage
        let depthData: AVDepthData?
        let timestamp: CMTime
        let faceDetected: Bool
        let exposureSettings: ExposureInfo
        
        internal init(rgbImage: CIImage, depthData: AVDepthData?, timestamp: CMTime, faceDetected: Bool, exposureSettings: ExposureInfo) {
            self.rgbImage = rgbImage
            self.depthData = depthData
            self.timestamp = timestamp
            self.faceDetected = faceDetected
            self.exposureSettings = exposureSettings
        }
    }
    
    /// Información de exposición
    internal struct ExposureInfo {
        let duration: CMTime
        let iso: Float
        let brightness: Float
        let isAdjusting: Bool
        
        internal init(duration: CMTime, iso: Float, brightness: Float, isAdjusting: Bool) {
            self.duration = duration
            self.iso = iso
            self.brightness = brightness
            self.isAdjusting = isAdjusting
        }
    }
    
    /// Errores específicos de la cámara
    internal enum CameraError: Error, LocalizedError {
        case trueDepthNotAvailable
        case permissionDenied
        case configurationFailed(String)
        case captureSessionError(String)
        case deviceNotFound
        case formatNotSupported
        
        internal var errorDescription: String? {
            switch self {
            case .trueDepthNotAvailable:
                return "TrueDepth camera not available on this device"
            case .permissionDenied:
                return "Camera permission denied"
            case .configurationFailed(let details):
                return "Camera configuration failed: \(details)"
            case .captureSessionError(let details):
                return "Capture session error: \(details)"
            case .deviceNotFound:
                return "Camera device not found"
            case .formatNotSupported:
                return "Camera format not supported"
            }
        }
    }
    
    // MARK: - Delegate Protocol
    
    internal protocol TrueDepthCameraDelegate: AnyObject {
        /// Frame procesado disponible
        func cameraManager(_ manager: TrueDepthCameraManager, didCapture frame: CapturedFrame)
        
        /// Estado de la cámara cambió
        func cameraManager(_ manager: TrueDepthCameraManager, didChangeState state: CameraState)
        
        /// Error en la cámara
        func cameraManager(_ manager: TrueDepthCameraManager, didEncounterError error: Error)
        
        /// Configuración de exposición cambió
        func cameraManager(_ manager: TrueDepthCameraManager, didUpdateExposure info: ExposureInfo)
    }
    
    // MARK: - Properties
    
    internal weak var delegate: TrueDepthCameraDelegate?
    
    private let configuration: CameraConfiguration
    private let sessionQueue = DispatchQueue(label: "camera.session.queue", qos: .userInitiated)
    private let processingQueue = DispatchQueue(label: "camera.processing.queue", qos: .userInteractive)
    
    private var captureSession: AVCaptureSession!
    private var frontCamera: AVCaptureDevice?
    private var videoInput: AVCaptureDeviceInput?
    private var videoOutput: AVCaptureVideoDataOutput?
    private var depthOutput: AVCaptureDepthDataOutput?
    private var outputSynchronizer: AVCaptureDataOutputSynchronizer?
    
    private var _state: CameraState = .notConfigured
    private let stateQueue = DispatchQueue(label: "camera.state.queue")
    
    internal var state: CameraState {
        return stateQueue.sync { _state }
    }
    
    private func setState(_ newState: CameraState) {
        stateQueue.sync {
            _state = newState
        }
        DispatchQueue.main.async {
            self.delegate?.cameraManager(self, didChangeState: newState)
        }
    }
    
    // MARK: - Initialization
    
    /// Inicializa el gestor con configuración específica
    /// - Parameter configuration: Configuración de la cámara
    internal init(configuration: CameraConfiguration = .default) {
        self.configuration = configuration
        super.init()
        setupCaptureSession()
    }
    
    deinit {
        stopCapture()
    }
    
    // MARK: - Public Methods
    
    /// Verifica si TrueDepth está disponible
    /// - Returns: true si el dispositivo tiene TrueDepth
    internal static func isTrueDepthAvailable() -> Bool {
        return AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) != nil
    }
    
    /// Solicita permisos de cámara
    /// - Returns: true si se concede el permiso
    internal func requestCameraPermission() async -> Bool {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        
        switch status {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .video)
        case .denied, .restricted:
            return false
        @unknown default:
            return false
        }
    }
    
    /// Configura la sesión de cámara
    internal func configure() async throws {
        guard await requestCameraPermission() else {
            throw CameraError.permissionDenied
        }
        
        setState(.configuring)
        
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            sessionQueue.async {
                do {
                    try self.configureCaptureSession()
                    self.setState(.configured)
                    continuation.resume()
                } catch {
                    self.setState(.failed(error))
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Inicia la captura de video
    internal func startCapture() throws {
        guard state == .configured || state == .stopped else {
            throw CameraError.configurationFailed("Camera not configured")
        }
        
        sessionQueue.async {
            self.captureSession.startRunning()
            self.setState(.running)
        }
    }
    
    /// Detiene la captura de video
    internal func stopCapture() {
        sessionQueue.async {
            self.captureSession.stopRunning()
            self.setState(.stopped)
        }
    }
    
    /// Obtiene preview layer para mostrar en UI
    /// - Returns: Layer para preview de la cámara
    internal func createPreviewLayer() -> AVCaptureVideoPreviewLayer {
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        return previewLayer
    }
    
    /// Configura manual de exposición
    /// - Parameters:
    ///   - duration: Duración de exposición
    ///   - iso: Valor ISO
    internal func setManualExposure(duration: CMTime, iso: Float) throws {
        guard let device = frontCamera else { return }
        
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        
        if device.isExposureModeSupported(.custom) {
            device.setExposureModeCustom(duration: duration, iso: iso) { _ in
                // Completion handler
            }
        }
    }
    
    /// Resetea exposición a automática
    internal func resetExposureToAuto() throws {
        guard let device = frontCamera else { return }
        
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        
        if device.isExposureModeSupported(.continuousAutoExposure) {
            device.exposureMode = .continuousAutoExposure
        }
    }
    
    /// Configura manual de enfoque
    /// - Parameter lensPosition: Posición del lente (0.0 - 1.0)
    internal func setManualFocus(lensPosition: Float) throws {
        guard let device = frontCamera else { return }
        
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        
        if device.isFocusModeSupported(.locked) {
            device.setFocusModeLocked(lensPosition: lensPosition) { _ in
                // Completion handler
            }
        }
    }
    
    /// Resetea enfoque a automático
    internal func resetFocusToAuto() throws {
        guard let device = frontCamera else { return }
        
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        
        if device.isFocusModeSupported(.continuousAutoFocus) {
            device.focusMode = .continuousAutoFocus
        }
    }
    
    // MARK: - Private Methods
    
    private func setupCaptureSession() {
        captureSession = AVCaptureSession()
    }
    
    private func configureCaptureSession() throws {
        captureSession.beginConfiguration()
        defer { captureSession.commitConfiguration() }
        
        // Configurar preset
        if captureSession.canSetSessionPreset(configuration.sessionPreset) {
            captureSession.sessionPreset = configuration.sessionPreset
        }
        
        // Configurar dispositivo de cámara
        try configureCameraDevice()
        
        // Configurar entrada de video
        try configureVideoInput()
        
        // Configurar salida de video
        try configureVideoOutput()
        
        // Configurar salida de depth (si está habilitada)
        if configuration.enableDepthCapture {
            try configureDepthOutput()
            try configureSynchronizer()
        }
    }
    
    private func configureCameraDevice() throws {
        // Buscar cámara TrueDepth frontal
        if let trueDepthCamera = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) {
            frontCamera = trueDepthCamera
        } else if let frontCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) {
            self.frontCamera = frontCamera
            print("⚠️ TrueDepth not available, using standard front camera")
        } else {
            throw CameraError.deviceNotFound
        }
        
        guard let camera = frontCamera else {
            throw CameraError.deviceNotFound
        }
        
        // Configurar dispositivo
        try camera.lockForConfiguration()
        defer { camera.unlockForConfiguration() }
        
        // Configurar frame rate
        let format = camera.activeFormat
        let ranges = format.videoSupportedFrameRateRanges
        
        if let range = ranges.first(where: {
            $0.minFrameRate <= Double(configuration.frameRate) &&
            Double(configuration.frameRate) <= $0.maxFrameRate
        }) {
            camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: configuration.frameRate)
            camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: configuration.frameRate)
        }
        
        // Configurar exposición
        if configuration.autoExposure && camera.isExposureModeSupported(.continuousAutoExposure) {
            camera.exposureMode = .continuousAutoExposure
        }
        
        // Configurar enfoque
        if configuration.autoFocus && camera.isFocusModeSupported(.continuousAutoFocus) {
            camera.focusMode = .continuousAutoFocus
        }
        
        // Configurar estabilización
        if camera.activeFormat.isVideoStabilizationModeSupported(.auto) {
            // La estabilización se configura en la conexión
        }
    }
    
    private func configureVideoInput() throws {
        guard let camera = frontCamera else {
            throw CameraError.deviceNotFound
        }
        
        let input = try AVCaptureDeviceInput(device: camera)
        
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
            videoInput = input
        } else {
            throw CameraError.configurationFailed("Cannot add video input")
        }
    }
    
    private func configureVideoOutput() throws {
        let output = AVCaptureVideoDataOutput()
        
        // Configurar formato de píxeles
        let pixelFormat = kCVPixelFormatType_32BGRA
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: pixelFormat
        ]
        
        // Configurar queue de procesamiento
        output.setSampleBufferDelegate(self, queue: processingQueue)
        
        // Descartar frames si el procesamiento es lento
        output.alwaysDiscardsLateVideoFrames = true
        
        if captureSession.canAddOutput(output) {
            captureSession.addOutput(output)
            videoOutput = output
            
            // Configurar estabilización en la conexión
            if let connection = output.connection(with: .video) {
                if connection.isVideoStabilizationSupported {
                    connection.preferredVideoStabilizationMode = .auto
                }
                
                // Configurar orientación
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
            }
        } else {
            throw CameraError.configurationFailed("Cannot add video output")
        }
    }
    
    private func configureDepthOutput() throws {
        guard Self.isTrueDepthAvailable() else {
            print("⚠️ Depth capture requested but TrueDepth not available")
            return
        }
        
        let output = AVCaptureDepthDataOutput()
        
        // Configurar delegate
        output.setDelegate(self, callbackQueue: processingQueue)
        
        // Configurar para filtrar datos de depth inválidos
        output.isFilteringEnabled = true
        
        if captureSession.canAddOutput(output) {
            captureSession.addOutput(output)
            depthOutput = output
            
            // Configurar conexión de depth
            if let connection = output.connection(with: .depthData) {
                connection.isEnabled = true
            }
        } else {
            throw CameraError.configurationFailed("Cannot add depth output")
        }
    }
    
    private func configureSynchronizer() throws {
        guard let videoOutput = videoOutput,
              let depthOutput = depthOutput else {
            return
        }
        
        // Crear sincronizador para RGB + Depth
        let synchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [videoOutput, depthOutput])
        synchronizer.setDelegate(self, queue: processingQueue)
        
        outputSynchronizer = synchronizer
    }
    
    private func processVideoFrame(_ sampleBuffer: CMSampleBuffer, depthData: AVDepthData? = nil) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let ciImage = CIImage(cvImageBuffer: imageBuffer)
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        
        // Extraer información de exposición
        let exposureInfo = extractExposureInfo()
        
        // Detectar rostro (simplificado para este ejemplo)
        let faceDetected = configuration.enableFaceDetection ? detectFaceInImage(ciImage) : true
        
        // Crear frame capturado
        let capturedFrame = CapturedFrame(
            rgbImage: ciImage,
            depthData: depthData,
            timestamp: timestamp,
            faceDetected: faceDetected,
            exposureSettings: exposureInfo
        )
        
        // Notificar al delegate
        DispatchQueue.main.async {
            self.delegate?.cameraManager(self, didCapture: capturedFrame)
        }
    }
    
    private func extractExposureInfo() -> ExposureInfo {
        guard let camera = frontCamera else {
            return ExposureInfo(duration: CMTime.zero, iso: 0, brightness: 0, isAdjusting: false)
        }
        
        return ExposureInfo(
            duration: camera.exposureDuration,
            iso: camera.iso,
            brightness: camera.lensAperture,
            isAdjusting: camera.isAdjustingExposure
        )
    }
    
    private func detectFaceInImage(_ image: CIImage) -> Bool {
        // Implementación simplificada - en producción usarías Vision framework
        // Por ahora retornamos true para simular detección
        return true
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension TrueDepthCameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    internal func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // Solo procesar si no hay sincronizador (modo solo RGB)
        guard outputSynchronizer == nil else { return }
        
        processVideoFrame(sampleBuffer)
    }
    
    internal func captureOutput(
        _ output: AVCaptureOutput,
        didDrop sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // Frame descartado - podríamos loggear para debugging
        print("📹 Frame dropped")
    }
}

// MARK: - AVCaptureDepthDataOutputDelegate

extension TrueDepthCameraManager: AVCaptureDepthDataOutputDelegate {
    
    internal func depthDataOutput(
        _ output: AVCaptureDepthDataOutput,
        didOutput depthData: AVDepthData,
        timestamp: CMTime,
        connection: AVCaptureConnection
    ) {
        // Solo procesar si no hay sincronizador
        guard outputSynchronizer == nil else { return }
        
        // En este caso tendríamos que mantener buffer de depth data
        // Por simplicidad, no implementamos esta lógica aquí
    }
}

// MARK: - AVCaptureDataOutputSynchronizerDelegate

extension TrueDepthCameraManager: AVCaptureDataOutputSynchronizerDelegate {
    
    internal func dataOutputSynchronizer(
        _ synchronizer: AVCaptureDataOutputSynchronizer,
        didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection
    ) {
        // Extraer datos sincronizados de RGB y depth
        guard let videoData = synchronizedDataCollection.synchronizedData(for: videoOutput!) as? AVCaptureSynchronizedSampleBufferData,
              !videoData.sampleBufferWasDropped else {
            return
        }
        
        let sampleBuffer = videoData.sampleBuffer
        
        // Extraer depth data si está disponible
        var depthData: AVDepthData?
        if let depthOutput = depthOutput,
           let depthSyncData = synchronizedDataCollection.synchronizedData(for: depthOutput) as? AVCaptureSynchronizedDepthData,
           !depthSyncData.depthDataWasDropped {
            depthData = depthSyncData.depthData
        }
        
        processVideoFrame(sampleBuffer, depthData: depthData)
    }
}
