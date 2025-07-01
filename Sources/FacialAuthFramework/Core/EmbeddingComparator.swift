// EmbeddingComparator.swift
// Sistema de comparación de embeddings faciales para autenticación 1:N

import Foundation
import Accelerate

/// Comparador de embeddings faciales con thresholds adaptativos
/// Implementa múltiples métricas de distancia y mejora continua
@available(iOS 14.0, *)
internal final class EmbeddingComparator {
    
    // MARK: - Types
    
    /// Métrica de distancia para comparación
    internal enum DistanceMetric {
        case cosine      // Distancia coseno (recomendada para embeddings normalizados)
        case euclidean   // Distancia euclidiana
        case manhattan   // Distancia de Manhattan
        case combined    // Combinación ponderada de métricas
    }
    
    /// Resultado de comparación entre embeddings
    internal struct ComparisonResult {
        let distance: Float
        let similarity: Float        // 0.0 = muy diferente, 1.0 = idéntico
        let isMatch: Bool
        let confidence: Float
        let metric: DistanceMetric
        let timestamp: Date
        
        internal init(distance: Float, similarity: Float, isMatch: Bool, confidence: Float, metric: DistanceMetric) {
            self.distance = distance
            self.similarity = similarity
            self.isMatch = isMatch
            self.confidence = confidence
            self.metric = metric
            self.timestamp = Date()
        }
    }
    
    /// Resultado de identificación 1:N
    internal struct IdentificationResult {
        let userID: String?
        let comparisonResult: ComparisonResult?
        let allComparisons: [String: ComparisonResult]  // userID -> resultado
        let processingTime: TimeInterval
        
        internal init(userID: String?, comparison: ComparisonResult?, allComparisons: [String: ComparisonResult], processingTime: TimeInterval) {
            self.userID = userID
            self.comparisonResult = comparison
            self.allComparisons = allComparisons
            self.processingTime = processingTime
        }
    }
    
    /// Configuración del comparador
    internal struct Configuration {
        let distanceMetric: DistanceMetric
        let baseThreshold: Float
        let adaptiveThresholds: Bool
        let confidenceWeighting: Bool
        let maxComparisons: Int          // Límite para performance en datasets grandes
        let minSimilarityGap: Float      // Gap mínimo entre mejor y segundo match
        
        internal static let `default` = Configuration(
            distanceMetric: .cosine,
            baseThreshold: 0.6,
            adaptiveThresholds: true,
            confidenceWeighting: true,
            maxComparisons: 1000,
            minSimilarityGap: 0.05
        )
        
        internal static let strict = Configuration(
            distanceMetric: .combined,
            baseThreshold: 0.8,
            adaptiveThresholds: true,
            confidenceWeighting: true,
            maxComparisons: 500,
            minSimilarityGap: 0.1
        )
        
        internal static let fast = Configuration(
            distanceMetric: .cosine,
            baseThreshold: 0.5,
            adaptiveThresholds: false,
            confidenceWeighting: false,
            maxComparisons: 100,
            minSimilarityGap: 0.03
        )
    }
    
    // MARK: - Properties
    
    private let configuration: Configuration
    private var userThresholds: [String: Float] = [:]
    private var userStatistics: [String: UserStatistics] = [:]
    private let statisticsQueue = DispatchQueue(label: "embedding.comparator.statistics", qos: .utility)
    
    // MARK: - Private Types
    
    private struct UserStatistics {
        var totalComparisons: Int
        var successfulMatches: Int
        var averageConfidence: Float
        var lastUpdated: Date
        var adaptiveThreshold: Float
        
        init(initialThreshold: Float) {
            self.totalComparisons = 0
            self.successfulMatches = 0
            self.averageConfidence = 0.0
            self.lastUpdated = Date()
            self.adaptiveThreshold = initialThreshold
        }
    }
    
    // MARK: - Initialization
    
    /// Inicializa el comparador con configuración específica
    /// - Parameter configuration: Configuración del comparador
    internal init(configuration: Configuration = .default) {
        self.configuration = configuration
    }
    
    // MARK: - Public Methods
    
    /// Compara dos embeddings individuales
    /// - Parameters:
    ///   - embedding1: Primer embedding a comparar
    ///   - embedding2: Segundo embedding a comparar
    ///   - userID: ID del usuario (opcional, para thresholds adaptativos)
    /// - Returns: Resultado de la comparación
    internal func compare(
        embedding1: [Float],
        embedding2: [Float],
        userID: String? = nil
    ) throws -> ComparisonResult {
        
        // Validar dimensiones
        guard embedding1.count == embedding2.count else {
            throw ComparatorError.dimensionMismatch(embedding1.count, embedding2.count)
        }
        
        guard !embedding1.isEmpty else {
            throw ComparatorError.emptyEmbedding
        }
        
        // Normalizar embeddings si es necesario
        let norm1 = normalizeEmbedding(embedding1)
        let norm2 = normalizeEmbedding(embedding2)
        
        // Calcular distancia según métrica configurada
        let distance = try calculateDistance(norm1, norm2, metric: configuration.distanceMetric)
        let similarity = distanceToSimilarity(distance, metric: configuration.distanceMetric)
        
        // Determinar threshold apropiado
        let threshold = getThreshold(for: userID)
        
        // Calcular confianza
        let confidence = calculateConfidence(similarity: similarity, threshold: threshold)
        
        // Determinar si es match
        let isMatch = similarity >= threshold && confidence >= 0.5
        
        // Actualizar estadísticas si se proporciona userID
        if let userID = userID {
            updateUserStatistics(userID: userID, similarity: similarity, isMatch: isMatch)
        }
        
        return ComparisonResult(
            distance: distance,
            similarity: similarity,
            isMatch: isMatch,
            confidence: confidence,
            metric: configuration.distanceMetric
        )
    }
    
    /// Identifica usuario en un conjunto de embeddings (1:N)
    /// - Parameters:
    ///   - queryEmbedding: Embedding a identificar
    ///   - userEmbeddings: Diccionario de userID -> [embeddings] registrados
    /// - Returns: Resultado de identificación con mejor match
    internal func identify(
        queryEmbedding: [Float],
        against userEmbeddings: [String: [[Float]]]
    ) async throws -> IdentificationResult {
        
        let startTime = Date()
        var allComparisons: [String: ComparisonResult] = [:]
        var bestMatch: (userID: String, result: ComparisonResult)?
        var comparisonsCount = 0
        
        // Limitar comparaciones para performance
        let maxUsers = min(userEmbeddings.count, configuration.maxComparisons / 5)
        let limitedUsers = Array(userEmbeddings.prefix(maxUsers))
        
        // Comparar contra todos los usuarios
        for (userID, embeddings) in limitedUsers {
            guard comparisonsCount < configuration.maxComparisons else { break }
            
            var bestUserResult: ComparisonResult?
            
            // Comparar contra todos los embeddings del usuario
            for embedding in embeddings {
                guard comparisonsCount < configuration.maxComparisons else { break }
                
                let comparison = try compare(
                    embedding1: queryEmbedding,
                    embedding2: embedding,
                    userID: userID
                )
                
                comparisonsCount += 1
                
                // Mantener la mejor comparación para este usuario
                if bestUserResult == nil || comparison.similarity > bestUserResult!.similarity {
                    bestUserResult = comparison
                }
            }
            
            // Guardar mejor resultado del usuario
            if let result = bestUserResult {
                allComparisons[userID] = result
                
                // Actualizar mejor match global
                if bestMatch == nil || result.similarity > bestMatch!.result.similarity {
                    bestMatch = (userID: userID, result: result)
                }
            }
        }
        
        // Validar que el mejor match cumple criterios de confianza
        let finalMatch = validateBestMatch(bestMatch, allComparisons: allComparisons)
        
        let processingTime = Date().timeIntervalSince(startTime)
        
        return IdentificationResult(
            userID: finalMatch?.userID,
            comparison: finalMatch?.result,
            allComparisons: allComparisons,
            processingTime: processingTime
        )
    }
    
    /// Actualiza threshold adaptativo para un usuario específico
    /// - Parameters:
    ///   - userID: Identificador del usuario
    ///   - newEmbeddings: Nuevos embeddings para recalcular threshold
    internal func updateAdaptiveThreshold(userID: String, newEmbeddings: [[Float]]) async {
        await withCheckedContinuation { continuation in
            statisticsQueue.async {
                // Calcular variabilidad interna de embeddings del usuario
                let internalVariability = self.calculateInternalVariability(embeddings: newEmbeddings)
                
                // Ajustar threshold basado en variabilidad
                let baseThreshold = self.configuration.baseThreshold
                let adaptedThreshold = self.adaptThresholdForVariability(
                    baseThreshold: baseThreshold,
                    variability: internalVariability
                )
                
                self.userThresholds[userID] = adaptedThreshold
                
                // Actualizar estadísticas
                if self.userStatistics[userID] == nil {
                    self.userStatistics[userID] = UserStatistics(initialThreshold: adaptedThreshold)
                }
                self.userStatistics[userID]?.adaptiveThreshold = adaptedThreshold
                self.userStatistics[userID]?.lastUpdated = Date()
                
                continuation.resume()
            }
        }
    }
    
    /// Obtiene estadísticas de un usuario
    /// - Parameter userID: Identificador del usuario
    /// - Returns: Estadísticas del usuario o nil si no existe
    internal func getUserStatistics(userID: String) -> (threshold: Float, totalComparisons: Int, successRate: Float)? {
        return statisticsQueue.sync {
            guard let stats = userStatistics[userID] else { return nil }
            
            let successRate = stats.totalComparisons > 0
                ? Float(stats.successfulMatches) / Float(stats.totalComparisons)
                : 0.0
            
            return (
                threshold: stats.adaptiveThreshold,
                totalComparisons: stats.totalComparisons,
                successRate: successRate
            )
        }
    }
    
    /// Resetea estadísticas de un usuario
    /// - Parameter userID: Identificador del usuario
    internal func resetUserStatistics(userID: String) {
        statisticsQueue.async {
            self.userStatistics.removeValue(forKey: userID)
            self.userThresholds.removeValue(forKey: userID)
        }
    }
    
    // MARK: - Private Methods
    
    private func normalizeEmbedding(_ embedding: [Float]) -> [Float] {
        // Calcular norma L2
        let norm = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
        
        // Evitar división por cero
        guard norm > Float.ulpOfOne else {
            return embedding
        }
        
        // Normalizar
        return embedding.map { $0 / norm }
    }
    
    private func calculateDistance(_ emb1: [Float], _ emb2: [Float], metric: DistanceMetric) throws -> Float {
        switch metric {
        case .cosine:
            return try calculateCosineDistance(emb1, emb2)
        case .euclidean:
            return try calculateEuclideanDistance(emb1, emb2)
        case .manhattan:
            return try calculateManhattanDistance(emb1, emb2)
        case .combined:
            return try calculateCombinedDistance(emb1, emb2)
        }
    }
    
    private func calculateCosineDistance(_ emb1: [Float], _ emb2: [Float]) throws -> Float {
        // Usar Accelerate para performance
        var dotProduct: Float = 0.0
        vDSP_dotpr(emb1, 1, emb2, 1, &dotProduct, vDSP_Length(emb1.count))
        
        // Cosine distance = 1 - cosine similarity
        // Como los embeddings están normalizados, dot product = cosine similarity
        return 1.0 - dotProduct
    }
    
    private func calculateEuclideanDistance(_ emb1: [Float], _ emb2: [Float]) throws -> Float {
        var squaredDistance: Float = 0.0
        
        // Calcular diferencias al cuadrado
        var differences = [Float](repeating: 0.0, count: emb1.count)
        vDSP_vsub(emb2, 1, emb1, 1, &differences, 1, vDSP_Length(emb1.count))
        vDSP_vsq(differences, 1, &differences, 1, vDSP_Length(differences.count))
        vDSP_sve(differences, 1, &squaredDistance, vDSP_Length(differences.count))
        
        return sqrt(squaredDistance)
    }
    
    private func calculateManhattanDistance(_ emb1: [Float], _ emb2: [Float]) throws -> Float {
        var distance: Float = 0.0
        
        // Calcular diferencias absolutas
        var differences = [Float](repeating: 0.0, count: emb1.count)
        vDSP_vsub(emb2, 1, emb1, 1, &differences, 1, vDSP_Length(emb1.count))
        vDSP_vabs(differences, 1, &differences, 1, vDSP_Length(differences.count))
        vDSP_sve(differences, 1, &distance, vDSP_Length(differences.count))
        
        return distance
    }
    
    private func calculateCombinedDistance(_ emb1: [Float], _ emb2: [Float]) throws -> Float {
        let cosineDistance = try calculateCosineDistance(emb1, emb2)
        let euclideanDistance = try calculateEuclideanDistance(emb1, emb2)
        
        // Normalizar euclidean distance
        let normalizedEuclidean = euclideanDistance / sqrt(Float(emb1.count))
        
        // Combinar con pesos
        return (cosineDistance * 0.7) + (normalizedEuclidean * 0.3)
    }
    
    private func distanceToSimilarity(_ distance: Float, metric: DistanceMetric) -> Float {
        switch metric {
        case .cosine:
            // Cosine distance -> similarity
            return 1.0 - distance
        case .euclidean, .manhattan:
            // Para distancias euclidianas/manhattan, usar función exponencial
            return exp(-distance)
        case .combined:
            // Combinado usa principalmente cosine, así que invertir
            return 1.0 - distance
        }
    }
    
    private func getThreshold(for userID: String?) -> Float {
        guard let userID = userID,
              configuration.adaptiveThresholds else {
            return configuration.baseThreshold
        }
        
        return statisticsQueue.sync {
            return userThresholds[userID] ?? configuration.baseThreshold
        }
    }
    
    private func calculateConfidence(similarity: Float, threshold: Float) -> Float {
        guard configuration.confidenceWeighting else {
            return similarity
        }
        
        // Confidence basado en qué tan por encima del threshold está
        let thresholdMargin = max(0, similarity - threshold)
        let maxMargin = 1.0 - threshold
        
        if maxMargin > 0 {
            return min(1.0, 0.5 + (thresholdMargin / maxMargin) * 0.5)
        } else {
            return similarity
        }
    }
    
    private func updateUserStatistics(userID: String, similarity: Float, isMatch: Bool) {
        statisticsQueue.async {
            if self.userStatistics[userID] == nil {
                self.userStatistics[userID] = UserStatistics(initialThreshold: self.configuration.baseThreshold)
            }
            
            guard var stats = self.userStatistics[userID] else { return }
            
            stats.totalComparisons += 1
            if isMatch {
                stats.successfulMatches += 1
            }
            
            // Actualizar promedio de confianza (moving average)
            let alpha: Float = 0.1
            stats.averageConfidence = (1 - alpha) * stats.averageConfidence + alpha * similarity
            stats.lastUpdated = Date()
            
            // Ajustar threshold adaptativamente cada 10 comparaciones
            if stats.totalComparisons % 10 == 0 {
                stats.adaptiveThreshold = self.adaptThresholdBasedOnHistory(stats)
            }
            
            self.userStatistics[userID] = stats
        }
    }
    
    private func adaptThresholdBasedOnHistory(_ stats: UserStatistics) -> Float {
        let successRate = Float(stats.successfulMatches) / Float(stats.totalComparisons)
        let baseThreshold = configuration.baseThreshold
        
        // Si el usuario tiene alta tasa de éxito, relajar threshold ligeramente
        // Si tiene baja tasa, hacerlo más estricto
        if successRate > 0.9 {
            return max(baseThreshold - 0.05, 0.3)
        } else if successRate < 0.7 {
            return min(baseThreshold + 0.05, 0.95)
        } else {
            return baseThreshold
        }
    }
    
    private func calculateInternalVariability(embeddings: [[Float]]) -> Float {
        guard embeddings.count > 1 else { return 0.0 }
        
        var totalDistance: Float = 0.0
        var comparisons = 0
        
        // Calcular distancia promedio entre todos los pares
        for i in 0..<embeddings.count {
            for j in (i+1)..<embeddings.count {
                if let distance = try? calculateCosineDistance(embeddings[i], embeddings[j]) {
                    totalDistance += distance
                    comparisons += 1
                }
            }
        }
        
        return comparisons > 0 ? totalDistance / Float(comparisons) : 0.0
    }
    
    private func adaptThresholdForVariability(baseThreshold: Float, variability: Float) -> Float {
        // Si el usuario tiene embeddings muy variables internamente, relajar threshold
        // Si son muy consistentes, puede ser más estricto
        let adjustment = variability * 0.2
        return max(0.3, min(0.95, baseThreshold - adjustment))
    }
    
    private func validateBestMatch(
        _ bestMatch: (userID: String, result: ComparisonResult)?,
        allComparisons: [String: ComparisonResult]
    ) -> (userID: String, result: ComparisonResult)? {
        
        guard let bestMatch = bestMatch else { return nil }
        
        // Verificar que el match es suficientemente bueno
        guard bestMatch.result.isMatch else { return nil }
        
        // Verificar gap mínimo con segundo mejor match
        let sortedResults = allComparisons.values.sorted { $0.similarity > $1.similarity }
        
        if sortedResults.count > 1 {
            let firstSimilarity = sortedResults[0].similarity
            let secondSimilarity = sortedResults[1].similarity
            let gap = firstSimilarity - secondSimilarity
            
            guard gap >= configuration.minSimilarityGap else {
                return nil // Muy cerca, no confiable
            }
        }
        
        return bestMatch
    }
}

// MARK: - Error Types

internal enum ComparatorError: Error, LocalizedError {
    case dimensionMismatch(Int, Int)
    case emptyEmbedding
    case calculationError(String)
    
    internal var errorDescription: String? {
        switch self {
        case .dimensionMismatch(let dim1, let dim2):
            return "Embedding dimension mismatch: \(dim1) vs \(dim2)"
        case .emptyEmbedding:
            return "Cannot compare empty embeddings"
        case .calculationError(let details):
            return "Distance calculation error: \(details)"
        }
    }
}
