// FaceDetectionManager.swift
// Sistema avanzado de detección y tracking facial continuo

import Foundation
import Vision
import CoreImage
import UIKit

/// Gestor de detección y tracking facial con Vision Framework
/// Mantiene continuidad de rostros entre frames y valida calidad facial
@available(iOS 14.0, *)
internal final class FaceDetectionManager {
    
    // MARK: - Types
    
    /// Rostro trackeado con historial
    internal struct TrackedFace {
        let id: UUID
        let currentObservation: VNFaceObservation
        let landmarks: VNFaceLandmarks2D?
        let quality: FaceQuality
        let trackingHistory: [TrackingEntry]
        let isStable: Bool
        let confidence: Float
        let lastUpdated: Date
        
        internal init(
            id: UUID = UUID(),
            observation: VNFaceObservation,
            landmarks: VNFaceLandmarks2D?,
            quality: FaceQuality,
            history: [TrackingEntry] = [],
            isStable: Bool = false,
            confidence: Float
        ) {
            self.id = id
            self.currentObservation = observation
            self.landmarks = landmarks
            self.quality = quality
            self.trackingHistory = history
            self.isStable = isStable
            self.confidence = confidence
            self.lastUpdated = Date()
        }
    }
    
    /// Entrada del historial de tracking
    internal struct TrackingEntry {
        let boundingBox: CGRect
        let timestamp: Date
        let confidence: Float
        
        internal init(boundingBox: CGRect, confidence: Float) {
            self.boundingBox = boundingBox
            self.timestamp = Date()
            self.confidence = confidence
        }
    }
    
    /// Calidad detallada del rostro
    internal struct FaceQuality {
        let size: Float              // Tamaño relativo (0.0 - 1.0)
        let position: Float          // Qué tan centrado está (0.0 - 1.0)
        let angle: Float             // Ángulo del rostro (0.0 - 1.0, 1.0 = frontal)
        let lighting: Float          // Calidad de iluminación (0.0 - 1.0)
        let sharpness: Float         // Nitidez del rostro (0.0 - 1.0)
        let expression: Float        // Expresión neutra (0.0 - 1.0)
        let eyesOpen: Float          // Ojos abiertos (0.0 - 1.0)
        let overall: Float           // Calidad general (0.0 - 1.0)
        let isGoodForAuth: Bool      // Si es apto para autenticación
        
        internal init(
            size: Float,
            position: Float,
            angle: Float,
            lighting: Float,
            sharpness: Float,
            expression: Float,
            eyesOpen: Float
        ) {
            self.size = size
            self.position = position
            self.angle = angle
            self.lighting = lighting
            self.sharpness = sharpness
            self.expression = expression
            self.eyesOpen = eyesOpen
            
            // Calcular calidad general con pesos específicos
            self.overall = (size * 0.2) + (position * 0.15) + (angle * 0.2) +
                          (lighting * 0.15) + (sharpness * 0.15) +
                          (expression * 0.05) + (eyesOpen * 0.1)
            
            // Criterios para autenticación
            self.isGoodForAuth = size > 0.3 &&
                                position > 0.6 &&
                                angle > 0.7 &&
                                lighting > 0.4 &&
                                sharpness > 0.5 &&
                                eyesOpen > 0.8 &&
                                overall > 0.6
        }
    }
    
    /// Guía visual para posicionamiento
    internal struct VisualGuide {
        let message: String
        let type: GuideType
        let targetArea: CGRect?        // Área objetivo para el rostro
        let currentArea: CGRect?       // Área actual del rostro
        let confidence: Float
        
        enum GuideType {
            case perfect           // "Perfecto"
            case moveCloser       // "Acércate"
            case moveFarther      // "Aléjate"
            case moveUp           // "Sube un poco"
            case moveDown         // "Baja un poco"
            case moveLeft         // "Muévete a la izquierda"
            case moveRight        // "Muévete a la derecha"
            case lookStraight     // "Mira directo a la cámara"
            case moreLight        // "Necesitas más luz"
            case holdStill        // "Mantente quieto"
            case openEyes         // "Abre los ojos"
        }
        
        internal init(message: String, type: GuideType, targetArea: CGRect? = nil, currentArea: CGRect? = nil, confidence: Float = 1.0) {
            self.message = message
            self.type = type
            self.targetArea = targetArea
            self.currentArea = currentArea
            self.confidence = confidence
        }
    }
    
    /// Configuración del detector
    internal struct Configuration {
        let maxTrackedFaces: Int                   // Máximo de rostros a trackear
        let trackingDistanceThreshold: Float       // Distancia máxima para considerar mismo rostro
        let stabilityRequiredFrames: Int           // Frames requeridos para considerar estable
        let qualityUpdateInterval: Int             // Cada cuántos frames actualizar calidad
        let enableLandmarkDetection: Bool          // Activar detección de landmarks
        let enableQualityValidation: Bool          // Activar validación de calidad
        let enableVisualGuides: Bool               // Activar guías visuales
        let targetFaceArea: CGRect                 // Área objetivo para el rostro
        
        internal static let `default` = Configuration(
            maxTrackedFaces: 1,
            trackingDistanceThreshold: 0.1,
            stabilityRequiredFrames: 5,
            qualityUpdateInterval: 3,
            enableLandmarkDetection: true,
            enableQualityValidation: true,
            enableVisualGuides: true,
            targetFaceArea: CGRect(x: 0.25, y: 0.3, width: 0.5, height: 0.6) // Centro de la imagen
        )
        
        internal static let strict = Configuration(
            maxTrackedFaces: 1,
            trackingDistanceThreshold: 0.05,
            stabilityRequiredFrames: 10,
            qualityUpdateInterval: 1,
            enableLandmarkDetection: true,
            enableQualityValidation: true,
            enableVisualGuides: true,
            targetFaceArea: CGRect(x: 0.3, y: 0.35, width: 0.4, height: 0.5)
        )
        
        internal static let relaxed = Configuration(
            maxTrackedFaces: 3,
            trackingDistanceThreshold: 0.2,
            stabilityRequiredFrames: 3,
            qualityUpdateInterval: 5,
            enableLandmarkDetection: false,
            enableQualityValidation: false,
            enableVisualGuides: false,
            targetFaceArea: CGRect(x: 0.2, y: 0.25, width: 0.6, height: 0.7)
        )
    }
    
    // MARK: - Delegate Protocol
    
    internal protocol FaceDetectionDelegate: AnyObject {
        /// Rostro trackeado actualizado
        func faceDetector(_ detector: FaceDetectionManager, didUpdateFace face: TrackedFace)
        
        /// Nuevo rostro detectado
        func faceDetector(_ detector: FaceDetectionManager, didDetectNewFace face: TrackedFace)
        
        /// Rostro perdido (salió del frame)
        func faceDetector(_ detector: FaceDetectionManager, didLoseFace faceID: UUID)
        
        /// Guía visual generada
        func faceDetector(_ detector: FaceDetectionManager, didGenerateGuide guide: VisualGuide)
        
        /// Error en detección
        func faceDetector(_ detector: FaceDetectionManager, didEncounterError error: Error)
    }
    
    // MARK: - Properties
    
    internal weak var delegate: FaceDetectionDelegate?
    
    private let configuration: Configuration
    private var trackedFaces: [UUID: TrackedFace] = [:]
    private var frameCounter: Int = 0
    
    // Vision requests
    private let faceDetectionRequest: VNDetectFaceRectanglesRequest
    private let faceLandmarksRequest: VNDetectFaceLandmarksRequest
    
    // Processing queues
    private let processingQueue = DispatchQueue(label: "face.detection.processing", qos: .userInteractive)
    private let trackingQueue = DispatchQueue(label: "face.tracking.queue", qos: .utility)
    
    // MARK: - Initialization
    
    /// Inicializa el detector con configuración específica
    /// - Parameter configuration: Configuración del detector
    internal init(configuration: Configuration = .default) {
        self.configuration = configuration
        self.faceDetectionRequest = VNDetectFaceRectanglesRequest()
        self.faceLandmarksRequest = VNDetectFaceLandmarksRequest()
        
        setupVisionRequests()
    }
    
    // MARK: - Public Methods
    
    /// Procesa una imagen para detectar y trackear rostros
    /// - Parameters:
    ///   - image: Imagen a procesar
    ///   - imageSize: Tamaño de la imagen original (para cálculos de posición)
    internal func processImage(_ image: CIImage, imageSize: CGSize) async throws {
        frameCounter += 1
        
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            processingQueue.async {
                do {
                    // Detectar rostros en la imagen
                    let detectedFaces = try self.detectFaces(in: image)
                    
                    // Procesar rostros detectados
                    self.processFaceDetections(detectedFaces, imageSize: imageSize)
                    
                    continuation.resume()
                } catch {
                    DispatchQueue.main.async {
                        self.delegate?.faceDetector(self, didEncounterError: error)
                    }
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Obtiene el rostro primario (mejor calidad y más estable)
    /// - Returns: Rostro primario o nil si no hay ninguno
    internal func getPrimaryFace() -> TrackedFace? {
        return trackingQueue.sync {
            return trackedFaces.values.sorted { face1, face2 in
                // Priorizar por estabilidad, luego por calidad
                if face1.isStable != face2.isStable {
                    return face1.isStable
                }
                return face1.quality.overall > face2.quality.overall
            }.first
        }
    }
    
    /// Obtiene todos los rostros trackeados
    /// - Returns: Array de rostros trackeados
    internal func getAllTrackedFaces() -> [TrackedFace] {
        return trackingQueue.sync {
            return Array(trackedFaces.values)
        }
    }
    
    /// Limpia rostros que no se han actualizado recientemente
    /// - Parameter timeoutInterval: Tiempo límite para considerar rostro perdido
    internal func cleanupLostFaces(timeoutInterval: TimeInterval = 1.0) {
        trackingQueue.async {
            let now = Date()
            let lostFaceIDs = self.trackedFaces.compactMap { (id, face) -> UUID? in
                return now.timeIntervalSince(face.lastUpdated) > timeoutInterval ? id : nil
            }
            
            for faceID in lostFaceIDs {
                self.trackedFaces.removeValue(forKey: faceID)
                DispatchQueue.main.async {
                    self.delegate?.faceDetector(self, didLoseFace: faceID)
                }
            }
        }
    }
    
    /// Resetea todos los rostros trackeados
    internal func resetTracking() {
        trackingQueue.sync {
            trackedFaces.removeAll()
            frameCounter = 0
        }
    }
    
    // MARK: - Private Methods
    
    private func setupVisionRequests() {
        // Configurar detección de rostros para máxima precisión
        if #available(iOS 15.0, *) {
            faceDetectionRequest.revision = VNDetectFaceRectanglesRequestRevision3
            faceLandmarksRequest.revision = VNDetectFaceLandmarksRequestRevision3
        } else {
            faceDetectionRequest.revision = VNDetectFaceRectanglesRequestRevision2
            faceLandmarksRequest.revision = VNDetectFaceLandmarksRequestRevision2
        }
    }
    
    private func detectFaces(in image: CIImage) throws -> [VNFaceObservation] {
        let handler = VNImageRequestHandler(ciImage: image)
        try handler.perform([faceDetectionRequest])
        return faceDetectionRequest.results ?? []
    }
    
    private func detectLandmarks(in image: CIImage, for faceObservation: VNFaceObservation) throws -> VNFaceLandmarks2D? {
        guard configuration.enableLandmarkDetection else { return nil }
        
        faceLandmarksRequest.inputFaceObservations = [faceObservation]
        
        let handler = VNImageRequestHandler(ciImage: image)
        try handler.perform([faceLandmarksRequest])
        
        return faceLandmarksRequest.results?.first?.landmarks
    }
    
    private func processFaceDetections(_ detections: [VNFaceObservation], imageSize: CGSize) {
        trackingQueue.async {
            var updatedFaceIDs: Set<UUID> = []
            
            for detection in detections {
                // Intentar asociar con rostro existente
                if let existingFaceID = self.findMatchingTrackedFace(for: detection) {
                    // Actualizar rostro existente
                    self.updateTrackedFace(id: existingFaceID, with: detection, imageSize: imageSize)
                    updatedFaceIDs.insert(existingFaceID)
                } else {
                    // Crear nuevo rostro trackeado
                    let newFace = self.createNewTrackedFace(from: detection, imageSize: imageSize)
                    self.trackedFaces[newFace.id] = newFace
                    updatedFaceIDs.insert(newFace.id)
                    
                    DispatchQueue.main.async {
                        self.delegate?.faceDetector(self, didDetectNewFace: newFace)
                    }
                }
            }
            
            // Limpiar rostros no actualizados (se han movido mucho o salido del frame)
            self.cleanupUnmatchedFaces(except: updatedFaceIDs)
        }
    }
    
    private func findMatchingTrackedFace(for detection: VNFaceObservation) -> UUID? {
        let detectionCenter = CGPoint(
            x: detection.boundingBox.midX,
            y: detection.boundingBox.midY
        )
        
        var bestMatch: (id: UUID, distance: Float)?
        
        for (id, trackedFace) in trackedFaces {
            let trackedCenter = CGPoint(
                x: trackedFace.currentObservation.boundingBox.midX,
                y: trackedFace.currentObservation.boundingBox.midY
            )
            
            let distance = Float(sqrt(
                pow(detectionCenter.x - trackedCenter.x, 2) +
                pow(detectionCenter.y - trackedCenter.y, 2)
            ))
            
            if distance <= configuration.trackingDistanceThreshold {
                if bestMatch == nil || distance < bestMatch!.distance {
                    bestMatch = (id: id, distance: distance)
                }
            }
        }
        
        return bestMatch?.id
    }
    
    private func updateTrackedFace(id: UUID, with detection: VNFaceObservation, imageSize: CGSize) {
        guard var trackedFace = trackedFaces[id] else { return }
        
        // Actualizar historial de tracking
        let trackingEntry = TrackingEntry(
            boundingBox: detection.boundingBox,
            confidence: detection.confidence
        )
        
        var newHistory = trackedFace.trackingHistory
        newHistory.append(trackingEntry)
        
        // Mantener solo los últimos N entries
        if newHistory.count > configuration.stabilityRequiredFrames * 2 {
            newHistory.removeFirst()
        }
        
        // Calcular estabilidad
        let isStable = calculateStability(from: newHistory)
        
        // Actualizar calidad si es necesario
        var quality = trackedFace.quality
        if frameCounter % configuration.qualityUpdateInterval == 0 && configuration.enableQualityValidation {
            quality = calculateFaceQuality(detection, imageSize: imageSize, landmarks: trackedFace.landmarks)
        }
        
        // Crear rostro actualizado
        let updatedFace = TrackedFace(
            id: id,
            observation: detection,
            landmarks: trackedFace.landmarks,
            quality: quality,
            history: newHistory,
            isStable: isStable,
            confidence: detection.confidence
        )
        
        trackedFaces[id] = updatedFace
        
        // Generar guía visual si está habilitada
        if configuration.enableVisualGuides {
            let guide = generateVisualGuide(for: updatedFace, imageSize: imageSize)
            DispatchQueue.main.async {
                self.delegate?.faceDetector(self, didGenerateGuide: guide)
            }
        }
        
        DispatchQueue.main.async {
            self.delegate?.faceDetector(self, didUpdateFace: updatedFace)
        }
    }
    
    private func createNewTrackedFace(from detection: VNFaceObservation, imageSize: CGSize) -> TrackedFace {
        // Calcular calidad inicial
        let quality = configuration.enableQualityValidation
            ? calculateFaceQuality(detection, imageSize: imageSize, landmarks: nil)
            : FaceQuality(size: 0.7, position: 0.7, angle: 0.7, lighting: 0.7, sharpness: 0.7, expression: 0.7, eyesOpen: 0.7)
        
        // Crear entrada inicial del historial
        let initialEntry = TrackingEntry(
            boundingBox: detection.boundingBox,
            confidence: detection.confidence
        )
        
        return TrackedFace(
            observation: detection,
            landmarks: nil,
            quality: quality,
            history: [initialEntry],
            isStable: false,
            confidence: detection.confidence
        )
    }
    
    private func calculateStability(from history: [TrackingEntry]) -> Bool {
        guard history.count >= configuration.stabilityRequiredFrames else { return false }
        
        let recentEntries = Array(history.suffix(configuration.stabilityRequiredFrames))
        let centers = recentEntries.map { entry in
            CGPoint(x: entry.boundingBox.midX, y: entry.boundingBox.midY)
        }
        
        // Calcular varianza de posición
        let avgX = centers.map { $0.x }.reduce(0, +) / CGFloat(centers.count)
        let avgY = centers.map { $0.y }.reduce(0, +) / CGFloat(centers.count)
        
        let variance = centers.map { center in
            pow(center.x - avgX, 2) + pow(center.y - avgY, 2)
        }.reduce(0, +) / CGFloat(centers.count)
        
        // Rostro estable si la varianza es baja
        return variance < 0.001 // Threshold de estabilidad
    }
    
    private func calculateFaceQuality(_ observation: VNFaceObservation, imageSize: CGSize, landmarks: VNFaceLandmarks2D?) -> FaceQuality {
        let boundingBox = observation.boundingBox
        
        // 1. Calcular tamaño relativo
        let faceArea = boundingBox.width * boundingBox.height
        let size = Float(min(faceArea * 4, 1.0)) // Normalizado para que 25% = 1.0
        
        // 2. Calcular posición relativa al target
        let faceCenter = CGPoint(x: boundingBox.midX, y: boundingBox.midY)
        let targetCenter = CGPoint(
            x: configuration.targetFaceArea.midX,
            y: configuration.targetFaceArea.midY
        )
        
        let distance = sqrt(
            pow(faceCenter.x - targetCenter.x, 2) +
            pow(faceCenter.y - targetCenter.y, 2)
        )
        let position = Float(max(0, 1.0 - (distance * 2)))
        
        // 3. Calcular ángulo (simplificado - en producción usarías landmarks)
        let angle: Float = 0.8 // Mock value
        
        // 4. Calcular iluminación (simplificado)
        let lighting: Float = 0.7 // Mock value
        
        // 5. Calcular nitidez (simplificado)
        let sharpness: Float = 0.8 // Mock value
        
        // 6. Calcular expresión neutra (simplificado)
        let expression: Float = 0.9 // Mock value
        
        // 7. Calcular ojos abiertos (simplificado)
        let eyesOpen: Float = 0.9 // Mock value
        
        return FaceQuality(
            size: size,
            position: position,
            angle: angle,
            lighting: lighting,
            sharpness: sharpness,
            expression: expression,
            eyesOpen: eyesOpen
        )
    }
    
    private func generateVisualGuide(for face: TrackedFace, imageSize: CGSize) -> VisualGuide {
        let boundingBox = face.currentObservation.boundingBox
        let quality = face.quality
        
        // Priorizar guías por importancia
        
        if quality.eyesOpen < 0.5 {
            return VisualGuide(message: "Abre los ojos", type: .openEyes, confidence: 1.0 - quality.eyesOpen)
        }
        
        if quality.lighting < 0.4 {
            return VisualGuide(message: "Necesitas más luz", type: .moreLight, confidence: quality.lighting)
        }
        
        if quality.size < 0.2 {
            return VisualGuide(message: "Acércate más", type: .moveCloser, targetArea: configuration.targetFaceArea, currentArea: boundingBox)
        }
        
        if quality.size > 0.8 {
            return VisualGuide(message: "Aléjate un poco", type: .moveFarther, targetArea: configuration.targetFaceArea, currentArea: boundingBox)
        }
        
        if quality.angle < 0.6 {
            return VisualGuide(message: "Mira directo a la cámara", type: .lookStraight, confidence: quality.angle)
        }
        
        if !face.isStable {
            return VisualGuide(message: "Mantente quieto", type: .holdStill, confidence: 0.5)
        }
        
        // Guías de posicionamiento
        let faceCenter = CGPoint(x: boundingBox.midX, y: boundingBox.midY)
        let targetCenter = CGPoint(x: configuration.targetFaceArea.midX, y: configuration.targetFaceArea.midY)
        
        let deltaX = faceCenter.x - targetCenter.x
        let deltaY = faceCenter.y - targetCenter.y
        
        if abs(deltaX) > 0.1 {
            if deltaX > 0 {
                return VisualGuide(message: "Muévete a la izquierda", type: .moveLeft, confidence: Float(abs(deltaX)))
            } else {
                return VisualGuide(message: "Muévete a la derecha", type: .moveRight, confidence: Float(abs(deltaX)))
            }
        }
        
        if abs(deltaY) > 0.1 {
            if deltaY > 0 {
                return VisualGuide(message: "Baja un poco", type: .moveDown, confidence: Float(abs(deltaY)))
            } else {
                return VisualGuide(message: "Sube un poco", type: .moveUp, confidence: Float(abs(deltaY)))
            }
        }
        
        if quality.isGoodForAuth && face.isStable {
            return VisualGuide(message: "¡Perfecto! Mantén la posición", type: .perfect, confidence: quality.overall)
        }
        
        return VisualGuide(message: "Ajusta tu posición", type: .lookStraight, confidence: quality.overall)
    }
    
    private func cleanupUnmatchedFaces(except updatedIDs: Set<UUID>) {
        let facesToRemove = trackedFaces.keys.filter { !updatedIDs.contains($0) }
        
        for faceID in facesToRemove {
            trackedFaces.removeValue(forKey: faceID)
            DispatchQueue.main.async {
                self.delegate?.faceDetector(self, didLoseFace: faceID)
            }
        }
        
        // Limitar número máximo de rostros trackeados
        if trackedFaces.count > configuration.maxTrackedFaces {
            let sortedFaces = trackedFaces.sorted { $0.value.quality.overall > $1.value.quality.overall }
            let facesToKeep = Array(sortedFaces.prefix(configuration.maxTrackedFaces))
            
            trackedFaces = Dictionary(uniqueKeysWithValues: facesToKeep)
        }
    }
}
