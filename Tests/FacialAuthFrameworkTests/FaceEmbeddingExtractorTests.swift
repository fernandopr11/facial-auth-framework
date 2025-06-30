import XCTest
import CoreImage
@testable import FacialAuthFramework

final class FaceEmbeddingExtractorTests: XCTestCase {
    
    var extractor: FaceEmbeddingExtractor?
    
    override func setUp() {
        super.setUp()
        
        // Intentar inicializar el extractor (fallará si no hay modelo)
        do {
            extractor = try FaceEmbeddingExtractor()
        } catch {
            // En tests sin modelo, el extractor será nil
            print("⚠️ FaceEmbeddingExtractor not available: \(error)")
            extractor = nil
        }
    }
    
    override func tearDown() {
        extractor = nil
        super.tearDown()
    }
    
    // MARK: - Configuration Tests
    
    func testConfigurationDefaults() {
        let config = FaceEmbeddingExtractor.Configuration.default
        
        XCTAssertEqual(config.modelName, "FaceRecognitionModel")
        XCTAssertEqual(config.minimumFaceQuality, 0.6)
        XCTAssertFalse(config.allowMultipleFaces)
        XCTAssertEqual(config.inputSize.width, 160)
        XCTAssertEqual(config.inputSize.height, 160)
    }
    
    func testCustomConfiguration() {
        let customConfig = FaceEmbeddingExtractor.Configuration(
            modelName: "CustomModel",
            minimumFaceQuality: 0.8,
            allowMultipleFaces: true,
            inputSize: CGSize(width: 224, height: 224)
        )
        
        XCTAssertEqual(customConfig.modelName, "CustomModel")
        XCTAssertEqual(customConfig.minimumFaceQuality, 0.8)
        XCTAssertTrue(customConfig.allowMultipleFaces)
        XCTAssertEqual(customConfig.inputSize.width, 224)
        XCTAssertEqual(customConfig.inputSize.height, 224)
    }
    
    // MARK: - Error Handling Tests
    
    func testExtractorInitializationWithMissingModel() {
        // Test que falla gracefully cuando no hay modelo
        let customConfig = FaceEmbeddingExtractor.Configuration(
            modelName: "NonExistentModel",
            minimumFaceQuality: 0.5,
            allowMultipleFaces: false,
            inputSize: CGSize(width: 160, height: 160)
        )
        
        XCTAssertThrowsError(try FaceEmbeddingExtractor(configuration: customConfig)) { error in
            guard let extractionError = error as? FaceEmbeddingExtractor.ExtractionError else {
                XCTFail("Expected ExtractionError")
                return
            }
            
            if case .modelNotFound = extractionError {
                // Expected error
            } else {
                XCTFail("Expected modelNotFound error")
            }
        }
    }
    
    func testExtractionErrorDescriptions() {
        let errors: [FaceEmbeddingExtractor.ExtractionError] = [
            .modelNotFound,
            .modelLoadFailed("Test error"),
            .noFaceDetected,
            .multipleFacesDetected,
            .faceQualityTooLow(0.3),
            .preprocessingFailed,
            .extractionFailed("Test failure")
        ]
        
        for error in errors {
            XCTAssertNotNil(error.errorDescription)
            XCTAssertFalse(error.errorDescription!.isEmpty)
        }
    }
    
    // MARK: - Face Quality Tests
    
    func testFaceQualityStructure() {
        let quality = FaceEmbeddingExtractor.FaceQuality(
            size: 0.8,
            sharpness: 0.7,
            illumination: 0.9,
            pose: 0.85
        )
        
        XCTAssertEqual(quality.size, 0.8)
        XCTAssertEqual(quality.sharpness, 0.7)
        XCTAssertEqual(quality.illumination, 0.9)
        XCTAssertEqual(quality.pose, 0.85)
        
        // Test overall calculation
        let expectedOverall = (0.8 * 0.3) + (0.7 * 0.25) + (0.9 * 0.25) + (0.85 * 0.2)
        XCTAssertEqual(quality.overall, Float(expectedOverall), accuracy: 0.001)
    }
    
    func testExtractionResultStructure() {
        let quality = FaceEmbeddingExtractor.FaceQuality(
            size: 0.8,
            sharpness: 0.7,
            illumination: 0.9,
            pose: 0.85
        )
        
        let embedding: [Float] = Array(0..<512).map { Float($0) / 512.0 }
        
        let result = FaceEmbeddingExtractor.ExtractionResult(
            embedding: embedding,
            confidence: 0.85,
            processingTime: 0.1,
            faceQuality: quality
        )
        
        XCTAssertEqual(result.embedding.count, 512)
        XCTAssertEqual(result.confidence, 0.85)
        XCTAssertEqual(result.processingTime, 0.1)
        XCTAssertEqual(result.faceQuality.overall, quality.overall)
    }
    
    // MARK: - Integration Tests (only if model is available)
    
    func testExtractEmbeddingWithNoFace() async throws {
        guard let extractor = extractor else {
            throw XCTSkip("FaceEmbeddingExtractor not available (no model found)")
        }
        
        // Test con imagen sin rostro
        let noFaceImage = createImageWithoutFace()
        
        do {
            _ = try await extractor.extractEmbedding(from: noFaceImage)
            XCTFail("Should have thrown noFaceDetected error")
        } catch FaceEmbeddingExtractor.ExtractionError.noFaceDetected {
            // Expected error
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }
    
    func testExtractEmbeddingWithSyntheticFace() async throws {
        guard let extractor = extractor else {
            throw XCTSkip("FaceEmbeddingExtractor not available (no model found)")
        }
        
        // Test con imagen sintética que simula un rostro
        let faceImage = createSyntheticFaceImage()
        
        do {
            let result = try await extractor.extractEmbedding(from: faceImage)
            
            // Verificar estructura del resultado
            XCTAssertFalse(result.embedding.isEmpty)
            XCTAssertTrue(result.confidence >= 0.0 && result.confidence <= 1.0)
            XCTAssertTrue(result.processingTime >= 0.0)
            XCTAssertTrue(result.faceQuality.overall >= 0.0 && result.faceQuality.overall <= 1.0)
            
        } catch FaceEmbeddingExtractor.ExtractionError.noFaceDetected {
            // Esperado con imagen sintética simple
            print("ℹ️ No face detected in synthetic image (expected)")
        } catch FaceEmbeddingExtractor.ExtractionError.faceQualityTooLow(let quality) {
            // Esperado con imagen de baja calidad
            print("ℹ️ Face quality too low: \(quality) (expected)")
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }
    
    // MARK: - Performance Tests
    
    func testModelLoadingPerformance() throws {
        guard extractor != nil else {
            throw XCTSkip("FaceEmbeddingExtractor not available (no model found)")
        }
        
        measure {
            do {
                _ = try FaceEmbeddingExtractor()
            } catch {
                XCTFail("Model loading failed: \(error)")
            }
        }
    }
    
    // MARK: - Helper Methods
    
    private func createImageWithoutFace() -> CIImage {
        // Crear imagen con contenido pero sin rostro
        let size = CGSize(width: 640, height: 480)
        let color = CIColor(red: 0.5, green: 0.7, blue: 0.3, alpha: 1.0)
        return CIImage(color: color).cropped(to: CGRect(origin: .zero, size: size))
    }
    
    private func createSyntheticFaceImage() -> CIImage {
        // Crear imagen sintética que podría parecer un rostro
        let size = CGSize(width: 640, height: 480)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        
        guard let context = CGContext(
            data: nil,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return createImageWithoutFace()
        }
        
        // Fondo
        context.setFillColor(CGColor(red: 0.9, green: 0.9, blue: 0.9, alpha: 1.0))
        context.fill(CGRect(origin: .zero, size: size))
        
        // "Rostro" - óvalo con tono piel
        let faceRect = CGRect(x: 220, y: 140, width: 200, height: 280)
        context.setFillColor(CGColor(red: 0.8, green: 0.7, blue: 0.6, alpha: 1.0))
        context.fillEllipse(in: faceRect)
        
        // "Ojos" - círculos oscuros
        context.setFillColor(CGColor(red: 0.2, green: 0.2, blue: 0.2, alpha: 1.0))
        context.fillEllipse(in: CGRect(x: 260, y: 200, width: 25, height: 15))
        context.fillEllipse(in: CGRect(x: 355, y: 200, width: 25, height: 15))
        
        // "Nariz" - pequeño triángulo
        context.setFillColor(CGColor(red: 0.7, green: 0.6, blue: 0.5, alpha: 1.0))
        context.move(to: CGPoint(x: 320, y: 250))
        context.addLine(to: CGPoint(x: 310, y: 270))
        context.addLine(to: CGPoint(x: 330, y: 270))
        context.closePath()
        context.fillPath()
        
        // "Boca" - elipse horizontal
        context.setFillColor(CGColor(red: 0.6, green: 0.3, blue: 0.3, alpha: 1.0))
        context.fillEllipse(in: CGRect(x: 300, y: 320, width: 40, height: 12))
        
        guard let cgImage = context.makeImage() else {
            return createImageWithoutFace()
        }
        
        return CIImage(cgImage: cgImage)
    }
}
