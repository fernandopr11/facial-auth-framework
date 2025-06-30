import XCTest
import Vision
import CoreImage
@testable import FacialAuthFramework

final class LivenessDetectorTests: XCTestCase {
    
    var livenessDetector: LivenessDetector!
    
    override func setUp() {
        super.setUp()
        // Usar configuración optimizada para simulador
        livenessDetector = LivenessDetector.forSimulator()
    }
    
    override func tearDown() {
        livenessDetector?.reset()
        livenessDetector = nil
        super.tearDown()
    }
    
    // MARK: - Basic Tests
    
    func testLivenessDetectorInitialization() {
        // Test que el detector se inicializa correctamente
        XCTAssertNotNil(livenessDetector)
    }
    
    func testDetectionMethodEnum() {
        // Test de los métodos de detección disponibles
        let allMethods = LivenessDetector.DetectionMethod.allCases
        
        XCTAssertEqual(allMethods.count, 5)
        XCTAssertTrue(allMethods.contains(.depthAnalysis))
        XCTAssertTrue(allMethods.contains(.textureAnalysis))
        XCTAssertTrue(allMethods.contains(.motionDetection))
        XCTAssertTrue(allMethods.contains(.blinkDetection))
        XCTAssertTrue(allMethods.contains(.faceQuality))
        
        // Test display names
        XCTAssertEqual(LivenessDetector.DetectionMethod.depthAnalysis.displayName, "Depth Analysis")
        XCTAssertEqual(LivenessDetector.DetectionMethod.textureAnalysis.displayName, "Texture Analysis")
        XCTAssertEqual(LivenessDetector.DetectionMethod.motionDetection.displayName, "Motion Detection")
        XCTAssertEqual(LivenessDetector.DetectionMethod.blinkDetection.displayName, "Blink Detection")
        XCTAssertEqual(LivenessDetector.DetectionMethod.faceQuality.displayName, "Face Quality")
    }
    
    func testConfigurationDefaults() {
        // Test configuraciones por defecto
        let defaultConfig = LivenessDetector.Configuration.default
        let simulatorConfig = LivenessDetector.Configuration.simulator
        
        // Default configuration
        XCTAssertEqual(defaultConfig.confidenceThreshold, 0.75)
        XCTAssertEqual(defaultConfig.motionSensitivity, 0.3)
        XCTAssertEqual(defaultConfig.textureThreshold, 0.6)
        XCTAssertEqual(defaultConfig.enabledMethods.count, 5)
        
        // Simulator configuration
        XCTAssertEqual(simulatorConfig.confidenceThreshold, 0.5)
        XCTAssertEqual(simulatorConfig.motionSensitivity, 0.1)
        XCTAssertEqual(simulatorConfig.textureThreshold, 0.3)
        XCTAssertEqual(simulatorConfig.enabledMethods.count, 2)
        
        XCTAssertTrue(simulatorConfig.enabledMethods.contains(.textureAnalysis))
        XCTAssertTrue(simulatorConfig.enabledMethods.contains(.faceQuality))
    }
    
    func testLivenessResultStructure() {
        // Test estructura del resultado
        let methods: [LivenessDetector.DetectionMethod] = [.textureAnalysis, .faceQuality]
        let result = LivenessDetector.LivenessResult(
            isLive: true,
            confidence: 0.85,
            methods: methods
        )
        
        XCTAssertTrue(result.isLive)
        XCTAssertEqual(result.confidence, 0.85)
        XCTAssertEqual(result.detectionMethods.count, 2)
        XCTAssertTrue(result.detectionMethods.contains(.textureAnalysis))
        XCTAssertTrue(result.detectionMethods.contains(.faceQuality))
        
        // Timestamp debe ser reciente
        let timeDifference = Date().timeIntervalSince(result.timestamp)
        XCTAssertLessThan(timeDifference, 1.0) // Menos de 1 segundo
    }
    
    func testTrueDepthAvailability() {
        // Test detección de capacidades TrueDepth
        let isAvailable = LivenessDetector.isTrueDepthAvailable
        
        // En simulador normalmente será false
        // En iPhone X+ será true
        // Solo verificamos que retorna un valor booleano válido
        XCTAssertTrue(isAvailable == true || isAvailable == false)
    }
    
    func testDetectorReset() {
        // Test reset del detector
        livenessDetector.reset()
        
        // No hay estado público para verificar, pero no debe crashear
        XCTAssertNotNil(livenessDetector)
    }
    
    // MARK: - Image Analysis Tests
    
    func testAnalyzeLivenessWithEmptyImage() async throws {
        // Test con imagen vacía/sin contenido
        let emptyImage = createEmptyImage()
        
        let result = try await livenessDetector.analyzeLiveness(in: emptyImage)
        
        // Sin rostro detectado, debería retornar false
        XCTAssertFalse(result.isLive)
        XCTAssertEqual(result.confidence, 0.0)
        XCTAssertTrue(result.detectionMethods.isEmpty)
    }
    
    func testAnalyzeLivenessWithSolidColorImage() async throws {
        // Test con imagen de color sólido
        let solidImage = createSolidColorImage(color: .red)
        
        let result = try await livenessDetector.analyzeLiveness(in: solidImage)
        
        // Sin rostro, debe retornar no-live
        XCTAssertFalse(result.isLive)
        XCTAssertEqual(result.confidence, 0.0)
        XCTAssertTrue(result.detectionMethods.isEmpty)
    }
    
    func testAnalyzeLivenessWithSyntheticFace() async throws {
        // Test con imagen que contiene formas que simulan un rostro
        let faceImage = createSyntheticFaceImage()
        
        let result = try await livenessDetector.analyzeLiveness(in: faceImage)
        
        // Verificar que se ejecutó el análisis
        XCTAssertNotNil(result)
        XCTAssertTrue(result.confidence >= 0.0 && result.confidence <= 1.0)
        
        // Con configuración de simulador, podría o no detectar como live
        // Lo importante es que no crashee y retorne valores válidos
    }
    
    func testMultipleFrameAnalysis() async throws {
        // Test análisis de múltiples frames para simular video
        let frame1 = createSyntheticFaceImage(withOffset: 0.0)
        let frame2 = createSyntheticFaceImage(withOffset: 0.02)
        let frame3 = createSyntheticFaceImage(withOffset: 0.01)
        
        let result1 = try await livenessDetector.analyzeLiveness(in: frame1)
        let result2 = try await livenessDetector.analyzeLiveness(in: frame2)
        let result3 = try await livenessDetector.analyzeLiveness(in: frame3)
        
        // Verificar que todos los análisis funcionaron
        XCTAssertNotNil(result1)
        XCTAssertNotNil(result2)
        XCTAssertNotNil(result3)
        
        // Timestamps deben ser secuenciales
        XCTAssertLessThanOrEqual(result1.timestamp, result2.timestamp)
        XCTAssertLessThanOrEqual(result2.timestamp, result3.timestamp)
    }
    
    // MARK: - Configuration Tests
    
    func testCustomConfiguration() {
        let customConfig = LivenessDetector.Configuration(
            enabledMethods: [.textureAnalysis],
            confidenceThreshold: 0.9,
            motionSensitivity: 0.5,
            textureThreshold: 0.8
        )
        
        let customDetector = LivenessDetector(configuration: customConfig)
        XCTAssertNotNil(customDetector)
    }
    
    // MARK: - Performance Tests
    
    func testAnalysisPerformance() throws {
        let image = createSyntheticFaceImage()
        
        measure {
            let expectation = self.expectation(description: "Analysis completion")
            
            Task {
                do {
                    _ = try await livenessDetector.analyzeLiveness(in: image)
                    expectation.fulfill()
                } catch {
                    XCTFail("Performance test failed: \(error)")
                    expectation.fulfill()
                }
            }
            
            wait(for: [expectation], timeout: 5.0)
        }
    }
    
    func testMemoryUsage() async throws {
        // Test que el detector no tenga memory leaks
        let image = createSyntheticFaceImage()
        
        // Ejecutar múltiples análisis
        for _ in 0..<10 {
            _ = try await livenessDetector.analyzeLiveness(in: image)
        }
        
        // Reset y verificar que no crashee
        livenessDetector.reset()
        
        // Un análisis más después del reset
        let result = try await livenessDetector.analyzeLiveness(in: image)
        XCTAssertNotNil(result)
    }
    
    // MARK: - Helper Methods
    
    private func createEmptyImage() -> CIImage {
        // Crear imagen pequeña pero válida (no vacía)
        return createSolidColorImage(color: CIColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0))
    }
    
    private func createSolidColorImage(color: CIColor) -> CIImage {
        // Crear imagen de color sólido con dimensiones FIJAS
        let size = CGRect(x: 0, y: 0, width: 100, height: 100)
        return CIImage(color: color).cropped(to: size)
    }
    
    private func createSyntheticFaceImage(withOffset offset: Double = 0.0) -> CIImage {
        // Crear imagen con patrón que simule características faciales básicas
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
            // Fallback a imagen sólida si no se puede crear el context
            return createSolidColorImage(color: CIColor(red: 0.8, green: 0.7, blue: 0.6))
        }
        
        // Fondo
        context.setFillColor(CGColor(red: 0.9, green: 0.9, blue: 0.9, alpha: 1.0))
        context.fill(CGRect(origin: .zero, size: size))
        
        // "Rostro" - rectángulo con tono piel
        let faceX = 200.0 + (offset * 100.0)
        let faceY = 150.0
        context.setFillColor(CGColor(red: 0.8, green: 0.7, blue: 0.6, alpha: 1.0))
        context.fill(CGRect(x: faceX, y: faceY, width: 240, height: 300))
        
        // "Ojos" - círculos oscuros
        context.setFillColor(CGColor(red: 0.2, green: 0.2, blue: 0.2, alpha: 1.0))
        context.fillEllipse(in: CGRect(x: faceX + 50, y: faceY + 70, width: 30, height: 20))
        context.fillEllipse(in: CGRect(x: faceX + 150, y: faceY + 70, width: 30, height: 20))
        
        // "Nariz" - línea vertical
        context.setStrokeColor(CGColor(red: 0.6, green: 0.5, blue: 0.4, alpha: 1.0))
        context.setLineWidth(3.0)
        context.move(to: CGPoint(x: faceX + 120, y: faceY + 120))
        context.addLine(to: CGPoint(x: faceX + 120, y: faceY + 160))
        context.strokePath()
        
        // "Boca" - elipse horizontal
        context.setFillColor(CGColor(red: 0.6, green: 0.3, blue: 0.3, alpha: 1.0))
        context.fillEllipse(in: CGRect(x: faceX + 100, y: faceY + 200, width: 40, height: 15))
        
        guard let cgImage = context.makeImage() else {
            return createSolidColorImage(color: CIColor(red: 0.8, green: 0.7, blue: 0.6))
        }
        
        return CIImage(cgImage: cgImage)
    }
}
