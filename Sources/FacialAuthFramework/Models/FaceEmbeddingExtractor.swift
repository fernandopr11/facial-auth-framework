// FaceEmbeddingExtractor.swift - Versión corregida para tu modelo específico

import Foundation
import CoreML
import Vision
import CoreImage

#if canImport(UIKit)
import UIKit
#endif

/// Extractor de embeddings faciales usando modelo InceptionResNetV1
/// Procesa imágenes y extrae características faciales
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
        case simulatorNotSupported
        
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
            case .simulatorNotSupported:
                return "Face recognition not supported in iOS Simulator. Please test on a physical device."
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
            modelName: "FaceRecognitionModel", // Tu modelo específico
            minimumFaceQuality: 0.6,
            allowMultipleFaces: false,
            inputSize: CGSize(width: 160, height: 160) // Según tu modelo: shape 160x160
        )
    }
    
    // MARK: - Initialization
    
    /// Inicializa el extractor con configuración específica
    /// - Parameter configuration: Configuración del extractor
    internal init(configuration: Configuration = .default) throws {
        print("🤖 === INICIALIZANDO FACE EMBEDDING EXTRACTOR ===")
        
        // Verificar si estamos en simulador
        #if targetEnvironment(simulator)
        print("⚠️ ADVERTENCIA: Ejecutándose en simulador iOS")
        print("⚠️ Vision Framework tiene problemas conocidos en simulador")
        print("⚠️ Para funcionalidad completa, usar dispositivo físico")
        #endif
        
        // Inicializar propiedades ANTES de usar self
        self.minimumFaceQuality = configuration.minimumFaceQuality
        self.inputSize = configuration.inputSize
        self.ciContext = CIContext(options: [.workingColorSpace: NSNull()])
        self.faceDetector = VNDetectFaceRectanglesRequest()
        
        // Cargar el modelo específico (tu FaceRecognitionModel.mlmodelc)
        print("🔍 Buscando modelo: '\(configuration.modelName).mlmodelc'")
        
        let frameworkBundle = Bundle.module
        
        guard let modelURL = frameworkBundle.url(forResource: configuration.modelName, withExtension: "mlmodelc") else {
            print("❌ Modelo no encontrado: \(configuration.modelName).mlmodelc")
            throw ExtractionError.modelNotFound
        }
        
        print("✅ Modelo encontrado: \(modelURL.path)")
        
        do {
            let modelConfiguration = MLModelConfiguration()
            
            // Configurar para máxima compatibilidad
            #if targetEnvironment(simulator)
            modelConfiguration.computeUnits = .cpuOnly // Solo CPU en simulador
            #else
            modelConfiguration.computeUnits = .all // CPU + GPU + Neural Engine en dispositivo
            #endif
            
            self.model = try MLModel(contentsOf: modelURL, configuration: modelConfiguration)
            print("✅ Modelo cargado exitosamente")
            
        } catch {
            print("❌ Error cargando modelo: \(error)")
            throw ExtractionError.modelLoadFailed(error.localizedDescription)
        }
        
        // Configurar detector de rostros DESPUÉS de inicializar todas las propiedades
        setupFaceDetector()
        
        // Ahora SÍ podemos llamar a debugModelInfo
        debugModelInfo()
        
        print("✅ FaceEmbeddingExtractor inicializado exitosamente")
    }
    
    // MARK: - Public Methods
    
    /// Extrae embedding facial de una imagen
    /// - Parameter image: Imagen que contiene un rostro
    /// - Returns: Resultado con embedding y metadatos
    internal func extractEmbedding(from image: CIImage) async throws -> ExtractionResult {
        print("🎯 === INICIANDO EXTRACCIÓN DE EMBEDDING ===")
        let startTime = Date()
        
        // Verificar simulador
        #if targetEnvironment(simulator)
        print("⚠️ Ejecutándose en simulador - usando funcionalidad limitada")
        return try await extractMockEmbedding(from: image, startTime: startTime)
        #else
        // En dispositivo real, usar funcionalidad completa
        return try await extractRealEmbedding(from: image, startTime: startTime)
        #endif
    }
    
    // MARK: - Private Methods
    
    private func debugModelInfo() {
        print("📋 === INFORMACIÓN DEL MODELO ===")
        let description = model.modelDescription
        
        print("📋 Metadatos:")
        for (key, value) in description.metadata {
            print("   \(key.rawValue): \(value)")
        }
        
        print("📥 Inputs:")
        for (name, desc) in description.inputDescriptionsByName {
            print("   \(name): \(desc.type)")
            if case .multiArray = desc.type {
                if let constraint = desc.multiArrayConstraint {
                    print("      Shape: \(constraint.shape)")
                    print("      Type: \(constraint.dataType)")
                }
            }
        }
        
        print("📤 Outputs:")
        for (name, desc) in description.outputDescriptionsByName {
            print("   \(name): \(desc.type)")
            if case .multiArray = desc.type {
                if let constraint = desc.multiArrayConstraint {
                    print("      Shape: \(constraint.shape)")
                    print("      Type: \(constraint.dataType)")
                }
            }
        }
    }
    
    private func extractMockEmbedding(from image: CIImage, startTime: Date) async throws -> ExtractionResult {
        print("🎭 Generando embedding mock para simulador...")
        
        // Simular procesamiento
        try await Task.sleep(nanoseconds: 100_000_000) // 0.1 segundos
        
        // Generar embedding mock realista
        // Tu modelo parece ser un modelo de embedding típico, así que generamos 512 dimensiones
        let embeddingSize = 512
        let mockEmbedding = (0..<embeddingSize).map { _ in Float.random(in: -1.0...1.0) }
        
        let mockQuality = FaceQuality(
            size: 0.8,
            sharpness: 0.7,
            illumination: 0.6,
            pose: 0.9
        )
        
        let processingTime = Date().timeIntervalSince(startTime)
        
        print("🎭 Mock embedding generado: \(mockEmbedding.count) dimensiones")
        
        return ExtractionResult(
            embedding: mockEmbedding,
            confidence: 0.85,
            processingTime: processingTime,
            faceQuality: mockQuality
        )
    }
    
    private func extractRealEmbedding(from image: CIImage, startTime: Date) async throws -> ExtractionResult {
        // 1. Detectar rostro en la imagen
        let faceObservation = try await detectSingleFace(in: image)
        print("✅ Rostro detectado: confidence=\(faceObservation.confidence)")
        
        // 2. Evaluar calidad del rostro
        let faceQuality = evaluateFaceQuality(observation: faceObservation, in: image)
        print("📊 Calidad del rostro: \(faceQuality.overall)")
        
        // 3. Verificar calidad mínima
        guard faceQuality.overall >= minimumFaceQuality else {
            throw ExtractionError.faceQualityTooLow(faceQuality.overall)
        }
        
        // 4. Recortar y preprocesar rostro
        let processedImage = try preprocessFace(observation: faceObservation, from: image)
        print("✅ Imagen preprocesada")
        
        // 5. Ejecutar modelo y extraer embedding
        let embedding = try await runModel(on: processedImage)
        print("✅ Embedding extraído: \(embedding.count) dimensiones")
        
        // 6. Calcular tiempo de procesamiento
        let processingTime = Date().timeIntervalSince(startTime)
        print("⏱️ Tiempo total: \(processingTime)s")
        
        return ExtractionResult(
            embedding: embedding,
            confidence: faceQuality.overall,
            processingTime: processingTime,
            faceQuality: faceQuality
        )
    }
    
    private func setupFaceDetector() {
        // Configurar para máxima precisión disponible
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
        let sizeScore = Float(min(faceArea * 4, 1.0))
        
        // 2-4. Valores simplificados para este ejemplo
        let sharpnessScore: Float = 0.75
        let illuminationScore: Float = 0.8
        let poseScore: Float = 0.85
        
        return FaceQuality(
            size: sizeScore,
            sharpness: sharpnessScore,
            illumination: illuminationScore,
            pose: poseScore
        )
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
        
        // Redimensionar al tamaño requerido por el modelo (160x160)
        let scaleX = inputSize.width / clampedRect.width
        let scaleY = inputSize.height / clampedRect.height
        let scaledImage = croppedFace.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        return scaledImage
    }
    
    private func runModel(on image: CIImage) async throws -> [Float] {
        print("🤖 Ejecutando modelo ML...")
        
        return try await withCheckedThrowingContinuation { continuation in
            do {
                print("🔄 Convirtiendo imagen a MLMultiArray...")
                let multiArray = try convertImageToMLMultiArray(image)
                print("✅ Imagen convertida a MLMultiArray shape: \(multiArray.shape)")
                
                // Crear input para tu modelo específico
                // Tu modelo espera input llamado "input" como MLMultiArray
                guard let input = try? MLDictionaryFeatureProvider(dictionary: [
                    "input": MLFeatureValue(multiArray: multiArray)
                ]) else {
                    continuation.resume(throwing: ExtractionError.preprocessingFailed)
                    return
                }
                
                print("🔄 Ejecutando predicción...")
                let prediction = try model.prediction(from: input)
                print("✅ Predicción completada")
                
                // Extraer embedding de tu modelo específico
                // Tu modelo tiene output llamado "output"
                guard let embeddingOutput = prediction.featureValue(for: "output")?.multiArrayValue else {
                    continuation.resume(throwing: ExtractionError.extractionFailed("No se encontró output 'output' en el modelo"))
                    return
                }
                
                // Convertir MLMultiArray a [Float]
                let embedding = (0..<embeddingOutput.count).map { index in
                    Float(embeddingOutput[index].floatValue)
                }
                
                print("✅ Embedding extraído: \(embedding.count) dimensiones")
                print("📊 Primeros 5 valores: \(Array(embedding.prefix(5)))")
                
                continuation.resume(returning: embedding)
                
            } catch {
                print("❌ Error en runModel: \(error)")
                continuation.resume(throwing: ExtractionError.extractionFailed(error.localizedDescription))
            }
        }
    }
    
    private func convertImageToMLMultiArray(_ image: CIImage) throws -> MLMultiArray {
        // Tu modelo espera: [1, 3, 160, 160] = [batch_size, channels(RGB), height, width]
        let shape = [1, 3, 160, 160] as [NSNumber]
        
        guard let multiArray = try? MLMultiArray(shape: shape, dataType: .float32) else {
            throw ExtractionError.preprocessingFailed
        }
        
        // Redimensionar imagen a 160x160
        let resizedImage = image.transformed(by: CGAffineTransform(scaleX: 160.0 / image.extent.width, y: 160.0 / image.extent.height))
        
        // Convertir a CGImage para acceso a píxeles
        guard let cgImage = ciContext.createCGImage(resizedImage, from: CGRect(x: 0, y: 0, width: 160, height: 160)) else {
            throw ExtractionError.preprocessingFailed
        }
        
        // Extraer datos de píxeles
        guard let dataProvider = cgImage.dataProvider,
              let pixelData = dataProvider.data,
              let data = CFDataGetBytePtr(pixelData) else {
            throw ExtractionError.preprocessingFailed
        }
        
        let bytesPerPixel = 4 // RGBA
        let width = 160
        let height = 160
        
        // Convertir RGBA a RGB normalizado y reorganizar a formato [1, 3, 160, 160]
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = (y * width + x) * bytesPerPixel
                
                // Extraer valores RGB (ignorar A)
                let r = Float(data[pixelIndex]) / 255.0
                let g = Float(data[pixelIndex + 1]) / 255.0
                let b = Float(data[pixelIndex + 2]) / 255.0
                
                // Calcular índices para formato [1, 3, 160, 160]
                // batch=0, channel=0(R), y=y, x=x
                let rIndex = 0 * (3 * 160 * 160) + 0 * (160 * 160) + y * 160 + x
                let gIndex = 0 * (3 * 160 * 160) + 1 * (160 * 160) + y * 160 + x
                let bIndex = 0 * (3 * 160 * 160) + 2 * (160 * 160) + y * 160 + x
                
                // Asignar valores normalizados
                multiArray[rIndex] = NSNumber(value: r)
                multiArray[gIndex] = NSNumber(value: g)
                multiArray[bIndex] = NSNumber(value: b)
            }
        }
        
        print("✅ MLMultiArray creado: shape \(multiArray.shape), count \(multiArray.count)")
        return multiArray
    }
}
