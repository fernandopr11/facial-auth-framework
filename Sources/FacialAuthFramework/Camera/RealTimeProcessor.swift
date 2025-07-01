// RealTimeProcessor.swift
// Pipeline de procesamiento de video en tiempo real para autenticación facial

import Foundation
import CoreImage
import AVFoundation
import Vision

/// Procesador de frames en tiempo real con buffer circular y filtros de calidad
/// Optimizado para performance y feedback visual continuo
@available(iOS 14.0, *)
internal final class RealTimeProcessor {
    
    // MARK: - Types
    
    /// Estado del procesamiento
    internal enum ProcessingState: Equatable {
        case idle
        case processing
        case analyzing
        case completed
        case error(Error)
        
        // Implementar Equatable manualmente para el caso .error
        static func == (lhs: ProcessingState, rhs: ProcessingState) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle),
                 (.processing, .processing),
                 (.analyzing, .analyzing),
                 (.completed, .completed):
                return true
            case (.error, .error):
                return true // Simplificado: todos los errores se consideran iguales
            default:
                return false
            }
        }
    }
    
    /// Calidad del frame procesado
    internal struct FrameQuality {
        let brightness: Float        // 0.0 - 1.0
        let sharpness: Float        // 0.0 - 1.0
        let contrast: Float         // 0.0 - 1.0
        let faceSize: Float         // 0.0 - 1.0 (tamaño relativo del rostro)
        let stability: Float        // 0.0 - 1.0 (qué tan estable está el rostro)
        let overall: Float          // 0.0 - 1.0 (calidad general)
        let isAcceptable: Bool      // Si cumple criterios mínimos
        
        internal init(brightness: Float, sharpness: Float, contrast: Float, faceSize: Float, stability: Float) {
            self.brightness = brightness
            self.sharpness = sharpness
            self.contrast = contrast
            self.faceSize = faceSize
            self.stability = stability
            
            // Calcular calidad general ponderada
            self.overall = (brightness * 0.2) + (sharpness * 0.3) + (contrast * 0.2) + (faceSize * 0.2) + (stability * 0.1)
            
            // Criterios mínimos para aceptabilidad
            self.isAcceptable = brightness > 0.3 &&
                               sharpness > 0.4 &&
                               contrast > 0.3 &&
                               faceSize > 0.15 &&
                               overall > 0.4
        }
    }
    
    /// Frame procesado con metadatos
    internal struct ProcessedFrame {
        let originalFrame: TrueDepthCameraManager.CapturedFrame
        let processedImage: CIImage
        let quality: FrameQuality
        let faceDetections: [VNFaceObservation]
        let processingTime: TimeInterval
        let timestamp: Date
        
        internal init(
            originalFrame: TrueDepthCameraManager.CapturedFrame,
            processedImage: CIImage,
            quality: FrameQuality,
            faceDetections: [VNFaceObservation],
            processingTime: TimeInterval
        ) {
            self.originalFrame = originalFrame
            self.processedImage = processedImage
            self.quality = quality
            self.faceDetections = faceDetections
            self.processingTime = processingTime
            self.timestamp = Date()
        }
    }
    
    /// Configuración del procesador
    internal struct Configuration {
        let bufferSize: Int                    // Tamaño del buffer circular
        let qualityThreshold: Float           // Threshold mínimo de calidad
        let maxProcessingTime: TimeInterval   // Tiempo máximo de procesamiento por frame
        let enableFaceDetection: Bool         // Activar detección facial
        let enableQualityFilters: Bool        // Activar filtros de calidad
        let enableStabilityTracking: Bool     // Activar tracking de estabilidad
        let feedbackEnabled: Bool             // Activar feedback visual
        
        internal static let `default` = Configuration(
            bufferSize: 5,
            qualityThreshold: 0.4,
            maxProcessingTime: 0.1, // 100ms
            enableFaceDetection: true,
            enableQualityFilters: true,
            enableStabilityTracking: true,
            feedbackEnabled: true
        )
        
        internal static let performance = Configuration(
            bufferSize: 3,
            qualityThreshold: 0.3,
            maxProcessingTime: 0.05, // 50ms
            enableFaceDetection: true,
            enableQualityFilters: false,
            enableStabilityTracking: false,
            feedbackEnabled: false
        )
        
        internal static let quality = Configuration(
            bufferSize: 10,
            qualityThreshold: 0.6,
            maxProcessingTime: 0.2, // 200ms
            enableFaceDetection: true,
            enableQualityFilters: true,
            enableStabilityTracking: true,
            feedbackEnabled: true
        )
    }
    
    /// Feedback visual para el usuario
    internal struct VisualFeedback {
        let message: String
        let type: FeedbackType
        let shouldShowOverlay: Bool
        let overlayColor: UIColor
        let confidence: Float
        
        enum FeedbackType {
            case guidance       // "Acércate más"
            case warning        // "Poca luz"
            case success        // "Perfecto"
            case error          // "Error"
        }
        
        internal init(message: String, type: FeedbackType, shouldShowOverlay: Bool = false, overlayColor: UIColor = .clear, confidence: Float = 1.0) {
            self.message = message
            self.type = type
            self.shouldShowOverlay = shouldShowOverlay
            self.overlayColor = overlayColor
            self.confidence = confidence
        }
    }
    
    // MARK: - Delegate Protocol
    
    internal protocol RealTimeProcessorDelegate: AnyObject {
        /// Frame procesado disponible
        func processor(_ processor: RealTimeProcessor, didProcess frame: ProcessedFrame)
        
        /// Estado del procesamiento cambió
        func processor(_ processor: RealTimeProcessor, didChangeState state: ProcessingState)
        
        /// Feedback visual disponible
        func processor(_ processor: RealTimeProcessor, didGenerateFeedback feedback: VisualFeedback)
        
        /// Error en el procesamiento
        func processor(_ processor: RealTimeProcessor, didEncounterError error: Error)
    }
    
    // MARK: - Properties
    
    internal weak var delegate: RealTimeProcessorDelegate?
    
    private let configuration: Configuration
    private let processingQueue = DispatchQueue(label: "realtime.processor.queue", qos: .userInteractive)
    private let analysisQueue = DispatchQueue(label: "realtime.analysis.queue", qos: .utility)
    
    // Buffer circular para frames
    private var frameBuffer: [TrueDepthCameraManager.CapturedFrame] = []
    private let bufferQueue = DispatchQueue(label: "frame.buffer.queue", qos: .userInteractive)
    
    // Estado del procesador
    private var _state: ProcessingState = .idle
    private let stateQueue = DispatchQueue(label: "processor.state.queue")
    
    internal var state: ProcessingState {
        return stateQueue.sync { _state }
    }
    
    // Components para análisis
    private let ciContext: CIContext
    private let faceDetectionRequest: VNDetectFaceRectanglesRequest
    private var previousFaceRect: CGRect?
    private var stabilityHistory: [CGRect] = []
    
    // Métricas de performance
    private var frameCount: Int = 0
    private var totalProcessingTime: TimeInterval = 0
    private var droppedFrames: Int = 0
    
    // MARK: - Initialization
    
    /// Inicializa el procesador con configuración específica
    /// - Parameter configuration: Configuración del procesador
    internal init(configuration: Configuration = .default) {
        self.configuration = configuration
        self.ciContext = CIContext(options: [.workingColorSpace: NSNull()])
        self.faceDetectionRequest = VNDetectFaceRectanglesRequest()
        
        setupFaceDetection()
    }
    
    // MARK: - Public Methods
    
    /// Procesa un frame capturado
    /// - Parameter frame: Frame de la cámara
    internal func processFrame(_ frame: TrueDepthCameraManager.CapturedFrame) {
        // Verificar si estamos procesando demasiado lento
        guard state != .processing else {
            droppedFrames += 1
            return
        }
        
        setState(.processing)
        
        processingQueue.async {
            let startTime = Date()
            
            do {
                // Añadir al buffer circular
                self.addToBuffer(frame)
                
                // Verificar calidad del frame
                guard self.configuration.enableQualityFilters else {
                    // Si no hay filtros, procesar directamente
                    try self.processFrameDirectly(frame, startTime: startTime)
                    return
                }
                
                // Analizar calidad
                let quality = try self.analyzeFrameQuality(frame)
                
                // Generar feedback visual
                if self.configuration.feedbackEnabled {
                    let feedback = self.generateVisualFeedback(quality: quality, frame: frame)
                    DispatchQueue.main.async {
                        self.delegate?.processor(self, didGenerateFeedback: feedback)
                    }
                }
                
                // Verificar si el frame es aceptable
                guard quality.isAcceptable else {
                    self.setState(.idle)
                    return
                }
                
                // Procesar frame de calidad
                try self.processQualityFrame(frame, quality: quality, startTime: startTime)
                
            } catch {
                self.setState(.error(error))
                DispatchQueue.main.async {
                    self.delegate?.processor(self, didEncounterError: error)
                }
            }
        }
    }
    
    /// Obtiene estadísticas de performance
    /// - Returns: Estadísticas del procesador
    internal func getPerformanceStats() -> (
        totalFrames: Int,
        averageProcessingTime: TimeInterval,
        droppedFrames: Int,
        dropRate: Float
    ) {
        let avgTime = frameCount > 0 ? totalProcessingTime / Double(frameCount) : 0
        let dropRate = frameCount > 0 ? Float(droppedFrames) / Float(frameCount + droppedFrames) : 0
        
        return (
            totalFrames: frameCount,
            averageProcessingTime: avgTime,
            droppedFrames: droppedFrames,
            dropRate: dropRate
        )
    }
    
    /// Resetea estadísticas de performance
    internal func resetStats() {
        frameCount = 0
        totalProcessingTime = 0
        droppedFrames = 0
    }
    
    /// Obtiene el último frame de calidad del buffer
    /// - Returns: Mejor frame disponible
    internal func getBestFrameFromBuffer() -> TrueDepthCameraManager.CapturedFrame? {
        return bufferQueue.sync {
            // Retornar el frame más reciente que detectó rostro
            return frameBuffer.last { $0.faceDetected }
        }
    }
    
    /// Limpia el buffer de frames
    internal func clearBuffer() {
        bufferQueue.sync {
            frameBuffer.removeAll()
        }
    }
    
    // MARK: - Private Methods
    
    private func setState(_ newState: ProcessingState) {
        stateQueue.sync {
            _state = newState
        }
        DispatchQueue.main.async {
            self.delegate?.processor(self, didChangeState: newState)
        }
    }
    
    private func setupFaceDetection() {
        // Configurar detección de rostros para máxima precisión
        if #available(iOS 15.0, *) {
            faceDetectionRequest.revision = VNDetectFaceRectanglesRequestRevision3
        } else {
            faceDetectionRequest.revision = VNDetectFaceRectanglesRequestRevision2
        }
    }
    
    private func addToBuffer(_ frame: TrueDepthCameraManager.CapturedFrame) {
        bufferQueue.sync {
            frameBuffer.append(frame)
            
            // Mantener tamaño del buffer
            if frameBuffer.count > configuration.bufferSize {
                frameBuffer.removeFirst()
            }
        }
    }
    
    private func processFrameDirectly(_ frame: TrueDepthCameraManager.CapturedFrame, startTime: Date) throws {
        // Procesamiento básico sin filtros de calidad
        let faceDetections = try detectFacesSync(in: frame.rgbImage)
        
        let mockQuality = FrameQuality(
            brightness: 0.7,
            sharpness: 0.7,
            contrast: 0.7,
            faceSize: faceDetections.isEmpty ? 0.0 : 0.8,
            stability: 0.8
        )
        
        let processingTime = Date().timeIntervalSince(startTime)
        
        let processedFrame = ProcessedFrame(
            originalFrame: frame,
            processedImage: frame.rgbImage,
            quality: mockQuality,
            faceDetections: faceDetections,
            processingTime: processingTime
        )
        
        updateStats(processingTime: processingTime)
        setState(.completed)
        
        DispatchQueue.main.async {
            self.delegate?.processor(self, didProcess: processedFrame)
        }
        
        setState(.idle)
    }
    
    private func processQualityFrame(
        _ frame: TrueDepthCameraManager.CapturedFrame,
        quality: FrameQuality,
        startTime: Date
    ) throws {
        setState(.analyzing)
        
        // Detectar rostros si está habilitado
        var faceDetections: [VNFaceObservation] = []
        if configuration.enableFaceDetection {
            faceDetections = try detectFacesSync(in: frame.rgbImage)
        }
        
        // Aplicar filtros de mejora de imagen
        let enhancedImage = try enhanceImage(frame.rgbImage, quality: quality)
        
        let processingTime = Date().timeIntervalSince(startTime)
        
        let processedFrame = ProcessedFrame(
            originalFrame: frame,
            processedImage: enhancedImage,
            quality: quality,
            faceDetections: faceDetections,
            processingTime: processingTime
        )
        
        updateStats(processingTime: processingTime)
        setState(.completed)
        
        DispatchQueue.main.async {
            self.delegate?.processor(self, didProcess: processedFrame)
        }
        
        setState(.idle)
    }
    
    private func analyzeFrameQuality(_ frame: TrueDepthCameraManager.CapturedFrame) throws -> FrameQuality {
        let image = frame.rgbImage
        
        // Analizar brillo
        let brightness = calculateBrightness(image)
        
        // Analizar nitidez
        let sharpness = calculateSharpness(image)
        
        // Analizar contraste
        let contrast = calculateContrast(image)
        
        // Analizar tamaño del rostro
        let faceSize = try calculateFaceSize(image)
        
        // Analizar estabilidad
        let stability = calculateStability(faceSize: faceSize, image: image)
        
        return FrameQuality(
            brightness: brightness,
            sharpness: sharpness,
            contrast: contrast,
            faceSize: faceSize,
            stability: stability
        )
    }
    
    private func calculateBrightness(_ image: CIImage) -> Float {
        // Usar filtro de promedio de área
        let averageFilter = CIFilter(name: "CIAreaAverage")!
        averageFilter.setValue(image, forKey: kCIInputImageKey)
        averageFilter.setValue(CIVector(cgRect: image.extent), forKey: kCIInputExtentKey)
        
        guard let outputImage = averageFilter.outputImage else { return 0.5 }
        
        var bitmap = [UInt8](repeating: 0, count: 4)
        ciContext.render(outputImage, toBitmap: &bitmap, rowBytes: 4, bounds: CGRect(x: 0, y: 0, width: 1, height: 1), format: .RGBA8, colorSpace: nil)
        
        let brightness = (Float(bitmap[0]) + Float(bitmap[1]) + Float(bitmap[2])) / (3.0 * 255.0)
        return brightness
    }
    
    private func calculateSharpness(_ image: CIImage) -> Float {
        // Usar filtro Laplaciano para detectar bordes
        guard let laplacianFilter = CIFilter(name: "CIConvolution3X3") else { return 0.5 }
        
        // Kernel Laplaciano
        let laplacianKernel = CIVector(values: [0, -1, 0, -1, 4, -1, 0, -1, 0], count: 9)
        
        laplacianFilter.setValue(image, forKey: kCIInputImageKey)
        laplacianFilter.setValue(laplacianKernel, forKey: "inputWeights")
        
        guard let outputImage = laplacianFilter.outputImage else { return 0.5 }
        
        // Calcular varianza (indicador de nitidez)
        // Implementación simplificada
        return 0.7 // Mock value for now
    }
    
    private func calculateContrast(_ image: CIImage) -> Float {
        // Usar estadísticas de histograma
        // Implementación simplificada
        return 0.6 // Mock value for now
    }
    
    private func calculateFaceSize(_ image: CIImage) throws -> Float {
        let faces = try detectFacesSync(in: image)
        
        guard let face = faces.first else { return 0.0 }
        
        let faceArea = face.boundingBox.width * face.boundingBox.height
        return Float(min(faceArea * 4, 1.0)) // Normalizar para que 25% del área = 1.0
    }
    
    private func calculateStability(faceSize: Float, image: CIImage) -> Float {
        guard configuration.enableStabilityTracking else { return 0.8 }
        
        // Detectar rostro actual
        do {
            let faces = try detectFacesSync(in: image)
            guard let currentFace = faces.first else { return 0.0 }
            
            let currentRect = currentFace.boundingBox
            
            // Comparar con historial
            stabilityHistory.append(currentRect)
            if stabilityHistory.count > 5 {
                stabilityHistory.removeFirst()
            }
            
            guard stabilityHistory.count > 1 else { return 0.5 }
            
            // Calcular variabilidad en posición
            let positions = stabilityHistory.map { Float($0.midX) }
            let avgPosition = positions.reduce(0, +) / Float(positions.count)
            let variance = positions.map { powf($0 - avgPosition, 2) }.reduce(0, +) / Float(positions.count)
            
            // Convertir varianza a score de estabilidad (menos varianza = más estable)
            let stability = max(0, 1.0 - (variance * 10))
            return stability
            
        } catch {
            return 0.5
        }
    }
    
    private func detectFaces(in image: CIImage) async throws -> [VNFaceObservation] {
        return try await withCheckedThrowingContinuation { continuation in
            let handler = VNImageRequestHandler(ciImage: image)
            
            do {
                try handler.perform([faceDetectionRequest])
                let observations = faceDetectionRequest.results ?? []
                continuation.resume(returning: observations)
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
    
    private func detectFacesSync(in image: CIImage) throws -> [VNFaceObservation] {
        let handler = VNImageRequestHandler(ciImage: image)
        try handler.perform([faceDetectionRequest])
        return faceDetectionRequest.results ?? []
    }
    
    private func enhanceImage(_ image: CIImage, quality: FrameQuality) throws -> CIImage {
        var enhancedImage = image
        
        // Ajustar brillo si es necesario
        if quality.brightness < 0.4 {
            let brightnessFilter = CIFilter(name: "CIColorControls")!
            brightnessFilter.setValue(enhancedImage, forKey: kCIInputImageKey)
            brightnessFilter.setValue(0.2, forKey: kCIInputBrightnessKey)
            enhancedImage = brightnessFilter.outputImage ?? enhancedImage
        }
        
        // Ajustar contraste si es necesario
        if quality.contrast < 0.4 {
            let contrastFilter = CIFilter(name: "CIColorControls")!
            contrastFilter.setValue(enhancedImage, forKey: kCIInputImageKey)
            contrastFilter.setValue(1.2, forKey: kCIInputContrastKey)
            enhancedImage = contrastFilter.outputImage ?? enhancedImage
        }
        
        // Aplicar sharpening si es necesario
        if quality.sharpness < 0.5 {
            let sharpenFilter = CIFilter(name: "CIUnsharpMask")!
            sharpenFilter.setValue(enhancedImage, forKey: kCIInputImageKey)
            sharpenFilter.setValue(0.5, forKey: kCIInputIntensityKey)
            enhancedImage = sharpenFilter.outputImage ?? enhancedImage
        }
        
        return enhancedImage
    }
    
    private func generateVisualFeedback(quality: FrameQuality, frame: TrueDepthCameraManager.CapturedFrame) -> VisualFeedback {
        // Generar feedback basado en la calidad del frame
        
        if quality.brightness < 0.3 {
            return VisualFeedback(
                message: "Necesitas más luz",
                type: .warning,
                shouldShowOverlay: true,
                overlayColor: .systemYellow.withAlphaComponent(0.3)
            )
        }
        
        if quality.faceSize < 0.15 {
            return VisualFeedback(
                message: "Acércate más a la cámara",
                type: .guidance,
                shouldShowOverlay: true,
                overlayColor: .systemBlue.withAlphaComponent(0.2)
            )
        }
        
        if quality.faceSize > 0.8 {
            return VisualFeedback(
                message: "Aléjate un poco",
                type: .guidance,
                shouldShowOverlay: true,
                overlayColor: .systemBlue.withAlphaComponent(0.2)
            )
        }
        
        if quality.stability < 0.4 {
            return VisualFeedback(
                message: "Mantén la cabeza quieta",
                type: .guidance
            )
        }
        
        if quality.sharpness < 0.4 {
            return VisualFeedback(
                message: "Imagen borrosa, mantén quieto",
                type: .warning
            )
        }
        
        if quality.isAcceptable {
            return VisualFeedback(
                message: "Perfecto, mantén la posición",
                type: .success,
                shouldShowOverlay: true,
                overlayColor: .systemGreen.withAlphaComponent(0.2),
                confidence: quality.overall
            )
        }
        
        return VisualFeedback(
            message: "Ajusta tu posición",
            type: .guidance
        )
    }
    
    private func updateStats(processingTime: TimeInterval) {
        frameCount += 1
        totalProcessingTime += processingTime
    }
}

// MARK: - Extensions

#if canImport(UIKit)
import UIKit

extension UIColor {
    static let systemYellow = UIColor.systemYellow
    static let systemBlue = UIColor.systemBlue
    static let systemGreen = UIColor.systemGreen
}
#endif
