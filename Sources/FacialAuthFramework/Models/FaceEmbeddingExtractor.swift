// FaceEmbeddingExtractor.swift
// Wrapper para el modelo InceptionResNetV1 CoreML

import Foundation
import CoreML
import Vision
import CoreImage

#if canImport(UIKit)
import UIKit
#endif

/// Extractor de embeddings faciales usando modelo InceptionResNetV1
/// Procesa imágenes y extrae características faciales de 512 dimensiones
@available(iOS 14.0, *)
internal final class FaceEmbeddingExtractor {
    
    // MARK: - Types
    
    /// Resultado de la extracción de embedding
    internal struct ExtractionResult {
        let embedding: [Float]
        let confidence: Float
        let processingTime: TimeInterval
        let faceQuality: FaceQuality
        
        internal init(embedding: [Float], confidence: Float, processingTime: TimeInterval, faceQuality: FaceQuality) {
            self.embedding = embedding
            self.confidence = confidence
            self.processingTime = processingTime
            self.faceQuality = faceQuality
        }
    }
    
    /// Calidad del rostro detectado
    internal struct FaceQuality {
        let size: Float           // Tamaño relativo del rostro
        let sharpness: Float      // Nitidez de la imagen
        let illumination: Float   // Calidad de iluminación
        let pose: Float          // Ángulo del rostro
        let overall: Float       // Calidad general (0.0 - 1.0)
        
        internal init(size: Float, sharpness: Float, illumination: Float, pose: Float) {
            self.size = size
            self.sharpness = sharpness
            self.illumination = illumination
            self.pose = pose
            self.overall = (size * 0.3 + sharpness * 0.25 + illumination * 0.25 + pose * 0.2)
        }
    }
    
    /// Errores específicos del extractor
    internal enum ExtractionError: Error, LocalizedError {
        case modelNotFound
        case modelLoadFailed(String)
        case noFaceDetected
        case multipleFacesDetected
        case faceQualityTooLow(Float)
        case preprocessingFailed
        case extractionFailed(String)
        
        internal var errorDescription: String? {
            switch self {
            case .modelNotFound:
                return "Face recognition model not found in bundle"
            case .modelLoadFailed(let details):
                return "Failed to load model: \(details)"
            case .noFaceDetected:
                return "No face detected in image"
            case .multipleFacesDetected:
                return "Multiple faces detected, expected single face"
            case .faceQualityTooLow(let quality):
                return "Face quality too low: \(quality)"
            case .preprocessingFailed:
                return "Image preprocessing failed"
            case .extractionFailed(let details):
                return "Embedding extraction failed: \(details)"
            }
        }
    }
    
    // MARK: - Properties
    
    private let model: MLModel
    private let faceDetector: VNDetectFaceRectanglesRequest
    private let minimumFaceQuality: Float
    private let inputSize: CGSize
    private let ciContext: CIContext
    
    // MARK: - Configuration
    
    internal struct Configuration {
        let modelName: String
        let minimumFaceQuality: Float
        let allowMultipleFaces: Bool
        let inputSize: CGSize
        
        internal static let `default` = Configuration(
            modelName: "FaceRecognitionModel",
            minimumFaceQuality: 0.6,
            allowMultipleFaces: false,
            inputSize: CGSize(width: 160, height: 160) // InceptionResNetV1 standard
        )
    }
    
    // MARK: - Initialization
    
    /// Inicializa el extractor con configuración específica
    /// - Parameter configuration: Configuración del extractor
    internal init(configuration: Configuration = .default) throws {
        self.minimumFaceQuality = configuration.minimumFaceQuality
        self.inputSize = configuration.inputSize
        self.ciContext = CIContext(options: [.workingColorSpace: NSNull()])
        
        // DEBUGGING INTENSIVO - Cargar el modelo CoreML
        print("🔍 === DEBUGGING MODEL LOADING ===")
        print("🔍 Looking for model: '\(configuration.modelName).mlmodel'")
        
        // 1. Información de bundles
        let frameworkBundle = Bundle.module
        let mainBundle = Bundle.main
        
        print("🔍 Framework bundle: \(frameworkBundle.bundlePath)")
        print("🔍 Framework bundle ID: \(frameworkBundle.bundleIdentifier ?? "none")")
        print("🔍 Main bundle: \(mainBundle.bundlePath)")
        print("🔍 Main bundle ID: \(mainBundle.bundleIdentifier ?? "none")")
        
        // 2. Listar TODOS los recursos en framework bundle
        if let resourcePath = frameworkBundle.resourcePath {
            print("🔍 Framework resource path: \(resourcePath)")
            if FileManager.default.fileExists(atPath: resourcePath) {
                let files = (try? FileManager.default.contentsOfDirectory(atPath: resourcePath)) ?? []
                print("🔍 Framework resources (\(files.count)):")
                files.sorted().forEach { print("    📄 \($0)") }
            } else {
                print("❌ Framework resource path doesn't exist")
            }
        } else {
            print("❌ Framework bundle has no resource path")
        }
        
        // 3. Listar recursos en main bundle
        if let resourcePath = mainBundle.resourcePath {
            print("🔍 Main bundle resource path: \(resourcePath)")
            let files = (try? FileManager.default.contentsOfDirectory(atPath: resourcePath)) ?? []
            let mlmodels = files.filter { $0.hasSuffix(".mlmodel") || $0.hasSuffix(".mlmodelc") }
            print("🔍 Main bundle ML models (\(mlmodels.count)):")
            mlmodels.forEach { print("    🤖 \($0)") }
        }
        
        // 4. Intentar múltiples variaciones del nombre
        let possibleNames = [
            configuration.modelName,
            "FaceRecognitionModel",
            "facerecognitionmodel",
            configuration.modelName.lowercased()
        ]
        
        var modelURL: URL?
        var foundIn = ""
        
        // Buscar en framework bundle
        for name in possibleNames {
            if let url = frameworkBundle.url(forResource: name, withExtension: "mlmodel") {
                modelURL = url
                foundIn = "framework bundle with name '\(name)'"
                break
            }
            if let url = frameworkBundle.url(forResource: name, withExtension: "mlmodelc") {
                modelURL = url
                foundIn = "framework bundle with name '\(name)' (compiled)"
                break
            }
        }
        
        // Si no se encuentra, buscar en main bundle
        if modelURL == nil {
            for name in possibleNames {
                if let url = mainBundle.url(forResource: name, withExtension: "mlmodel") {
                    modelURL = url
                    foundIn = "main bundle with name '\(name)'"
                    break
                }
                if let url = mainBundle.url(forResource: name, withExtension: "mlmodelc") {
                    modelURL = url
                    foundIn = "main bundle with name '\(name)' (compiled)"
                    break
                }
            }
        }
        
        // 5. Resultado
        guard let finalModelURL = modelURL else {
            print("❌ MODEL NOT FOUND in any bundle with any name variation")
            print("🔍 Tried names: \(possibleNames)")
            throw ExtractionError.modelNotFound
        }
        
        print("✅ MODEL FOUND: \(finalModelURL.path)")
        print("✅ Found in: \(foundIn)")
        print("🔍 === END DEBUGGING ===")
        
        // Usar la URL encontrada para cargar el modelo
        
        do {
            let modelConfiguration = MLModelConfiguration()
            modelConfiguration.computeUnits = .all // CPU + GPU + Neural Engine
            self.model = try MLModel(contentsOf: finalModelURL, configuration: modelConfiguration)
        } catch {
            throw ExtractionError.modelLoadFailed(error.localizedDescription)
        }
        
        // Configurar detector de rostros
        self.faceDetector = VNDetectFaceRectanglesRequest()
        setupFaceDetector()
    }
    
    // MARK: - Public Methods
    
    /// Extrae embedding facial de una imagen
    /// - Parameter image: Imagen que contiene un rostro
    /// - Returns: Resultado con embedding y metadatos
    internal func extractEmbedding(from image: CIImage) async throws -> ExtractionResult {
        let startTime = Date()
        
        // 1. Detectar rostro en la imagen
        let faceObservation = try await detectSingleFace(in: image)
        
        // 2. Evaluar calidad del rostro
        let faceQuality = evaluateFaceQuality(observation: faceObservation, in: image)
        
        // 3. Verificar calidad mínima
        guard faceQuality.overall >= minimumFaceQuality else {
            throw ExtractionError.faceQualityTooLow(faceQuality.overall)
        }
        
        // 4. Recortar y preprocesar rostro
        let processedImage = try preprocessFace(observation: faceObservation, from: image)
        
        // 5. Ejecutar modelo y extraer embedding
        let embedding = try await runModel(on: processedImage)
        
        // 6. Calcular tiempo de procesamiento
        let processingTime = Date().timeIntervalSince(startTime)
        
        return ExtractionResult(
            embedding: embedding,
            confidence: faceQuality.overall,
            processingTime: processingTime,
            faceQuality: faceQuality
        )
    }
    
    /// Extrae embeddings de múltiples rostros (si está habilitado)
    /// - Parameter image: Imagen que puede contener múltiples rostros
    /// - Returns: Array de resultados, uno por rostro detectado
    internal func extractMultipleEmbeddings(from image: CIImage) async throws -> [ExtractionResult] {
        let startTime = Date()
        
        // Detectar todos los rostros
        let faceObservations = try await detectFaces(in: image)
        
        guard !faceObservations.isEmpty else {
            throw ExtractionError.noFaceDetected
        }
        
        var results: [ExtractionResult] = []
        
        // Procesar cada rostro detectado
        for observation in faceObservations {
            do {
                let faceQuality = evaluateFaceQuality(observation: observation, in: image)
                
                // Solo procesar rostros de calidad suficiente
                guard faceQuality.overall >= minimumFaceQuality else {
                    continue
                }
                
                let processedImage = try preprocessFace(observation: observation, from: image)
                let embedding = try await runModel(on: processedImage)
                
                let processingTime = Date().timeIntervalSince(startTime)
                
                let result = ExtractionResult(
                    embedding: embedding,
                    confidence: faceQuality.overall,
                    processingTime: processingTime,
                    faceQuality: faceQuality
                )
                
                results.append(result)
            } catch {
                // Continuar con el siguiente rostro si uno falla
                continue
            }
        }
        
        return results
    }
    
    // MARK: - Private Methods
    
    private func setupFaceDetector() {
        // Configurar para máxima precisión
        if #available(iOS 15.0, *) {
            faceDetector.revision = VNDetectFaceRectanglesRequestRevision3
        } else {
            faceDetector.revision = VNDetectFaceRectanglesRequestRevision2
        }
    }
    
    private func detectSingleFace(in image: CIImage) async throws -> VNFaceObservation {
        let faces = try await detectFaces(in: image)
        
        guard !faces.isEmpty else {
            throw ExtractionError.noFaceDetected
        }
        
        guard faces.count == 1 else {
            throw ExtractionError.multipleFacesDetected
        }
        
        return faces[0]
    }
    
    private func detectFaces(in image: CIImage) async throws -> [VNFaceObservation] {
        return try await withCheckedThrowingContinuation { continuation in
            let handler = VNImageRequestHandler(ciImage: image)
            
            do {
                try handler.perform([faceDetector])
                let observations = faceDetector.results ?? []
                continuation.resume(returning: observations)
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
    
    private func evaluateFaceQuality(observation: VNFaceObservation, in image: CIImage) -> FaceQuality {
        let boundingBox = observation.boundingBox
        
        // 1. Evaluar tamaño del rostro
        let faceArea = boundingBox.width * boundingBox.height
        let sizeScore = Float(min(faceArea * 4, 1.0)) // Normalizar para que 25% del área = score 1.0
        
        // 2. Evaluar nitidez (simplificado)
        let sharpnessScore = evaluateSharpness(in: image, region: boundingBox)
        
        // 3. Evaluar iluminación
        let illuminationScore = evaluateIllumination(in: image, region: boundingBox)
        
        // 4. Evaluar pose del rostro
        let poseScore = evaluatePose(observation: observation)
        
        return FaceQuality(
            size: sizeScore,
            sharpness: sharpnessScore,
            illumination: illuminationScore,
            pose: poseScore
        )
    }
    
    private func evaluateSharpness(in image: CIImage, region: CGRect) -> Float {
        // Implementación simplificada de evaluación de nitidez
        // En producción usarías análisis de gradientes o varianza de Laplaciano
        let imageExtent = image.extent
        let faceRegion = CGRect(
            x: region.minX * imageExtent.width,
            y: region.minY * imageExtent.height,
            width: region.width * imageExtent.width,
            height: region.height * imageExtent.height
        )
        
        let croppedImage = image.cropped(to: faceRegion)
        
        // Mock implementation - en producción calcularías la varianza de gradientes
        return 0.75 // Valor simulado
    }
    
    private func evaluateIllumination(in image: CIImage, region: CGRect) -> Float {
        // Evaluar si la iluminación es adecuada para reconocimiento
        // Mock implementation
        return 0.8 // Valor simulado
    }
    
    private func evaluatePose(observation: VNFaceObservation) -> Float {
        // Evaluar si el rostro está en un ángulo adecuado
        // En producción usarías face landmarks para calcular yaw, pitch, roll
        return 0.85 // Valor simulado
    }
    
    private func preprocessFace(observation: VNFaceObservation, from image: CIImage) throws -> CIImage {
        let imageExtent = image.extent
        let boundingBox = observation.boundingBox
        
        // Convertir coordenadas normalizadas a píxeles
        let faceRect = CGRect(
            x: boundingBox.minX * imageExtent.width,
            y: boundingBox.minY * imageExtent.height,
            width: boundingBox.width * imageExtent.width,
            height: boundingBox.height * imageExtent.height
        )
        
        // Expandir región para incluir más contexto facial
        let expandedRect = faceRect.insetBy(dx: -faceRect.width * 0.2, dy: -faceRect.height * 0.2)
        let clampedRect = expandedRect.intersection(imageExtent)
        
        // Recortar rostro
        let croppedFace = image.cropped(to: clampedRect)
        
        // Redimensionar al tamaño requerido por el modelo
        let scaleX = inputSize.width / clampedRect.width
        let scaleY = inputSize.height / clampedRect.height
        let scaledImage = croppedFace.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        // Normalizar píxeles para el modelo (0-1 range)
        // El modelo InceptionResNetV1 espera valores normalizados
        return scaledImage
    }
    
    private func runModel(on image: CIImage) async throws -> [Float] {
        return try await withCheckedThrowingContinuation { continuation in
            do {
                // Convertir CIImage a CVPixelBuffer para el modelo
                var pixelBuffer: CVPixelBuffer?
                let attributes: [String: Any] = [
                    kCVPixelBufferCGImageCompatibilityKey as String: true,
                    kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
                ]
                
                let status = CVPixelBufferCreate(
                    kCFAllocatorDefault,
                    Int(inputSize.width),
                    Int(inputSize.height),
                    kCVPixelFormatType_32ARGB,
                    attributes as CFDictionary,
                    &pixelBuffer
                )
                
                guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
                    throw ExtractionError.preprocessingFailed
                }
                
                // Renderizar imagen al buffer
                ciContext.render(image, to: buffer)
                
                // Crear input para el modelo
                guard let input = try? MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: buffer)]) else {
                    continuation.resume(throwing: ExtractionError.preprocessingFailed)
                    return
                }
                
                // Ejecutar predicción
                let prediction = try model.prediction(from: input)
                
                // Extraer embedding (ajustar el nombre del output según tu modelo)
                guard let embeddingOutput = prediction.featureValue(for: "embedding")?.multiArrayValue ??
                      prediction.featureValue(for: "output")?.multiArrayValue ??
                      prediction.featureValue(for: "features")?.multiArrayValue else {
                    continuation.resume(throwing: ExtractionError.extractionFailed("No embedding output found"))
                    return
                }
                
                // Convertir MLMultiArray a [Float]
                let embedding = (0..<embeddingOutput.count).map { index in
                    Float(embeddingOutput[index].floatValue)
                }
                
                continuation.resume(returning: embedding)
                
            } catch {
                continuation.resume(throwing: ExtractionError.extractionFailed(error.localizedDescription))
            }
        }
    }
}
