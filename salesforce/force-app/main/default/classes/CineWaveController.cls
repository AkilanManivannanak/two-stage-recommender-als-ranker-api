/**
 * Lightning controller for the cineWaveRecommendations LWC.
 *
 * `cacheable=true` on the read lets the Lightning Data Service cache the slate
 * client-side, so scrolling a record page does not re-hit a metered API. The
 * feedback write is deliberately NOT cacheable — it must reach the recommender
 * every time, because it is training signal.
 */
public with sharing class CineWaveController {

    @AuraEnabled(cacheable=true)
    public static List<CineWaveRecommendation> getRecommendations(Integer userId, Integer k) {
        try {
            return CineWaveGraphQLClient.getRecommendations(
                userId, (k == null || k <= 0) ? 6 : k);
        } catch (Exception e) {
            throw new AuraHandledException(e.getMessage());
        }
    }

    @AuraEnabled(cacheable=true)
    public static Map<String, Object> getModelMetrics() {
        try {
            return CineWaveGraphQLClient.getModelMetrics();
        } catch (Exception e) {
            throw new AuraHandledException(e.getMessage());
        }
    }

    @AuraEnabled
    public static Boolean recordFeedback(Integer userId, Integer itemId,
                                         String eventName, Decimal dwellSeconds) {
        try {
            return CineWaveGraphQLClient.recordFeedback(
                userId, itemId, eventName, dwellSeconds);
        } catch (Exception e) {
            throw new AuraHandledException(e.getMessage());
        }
    }
}
