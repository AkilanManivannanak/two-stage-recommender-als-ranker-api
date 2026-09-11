// Anonymous Apex end-to-end check against a live CineWave deployment.
// Run: sf apex run --file scripts/smoke.apex
CineWaveRecommenderAction.Request r = new CineWaveRecommenderAction.Request();
r.userId = 1;
r.numberOfTitles = 5;

List<CineWaveRecommenderAction.Result> out =
    CineWaveRecommenderAction.getRecommendations(
        new List<CineWaveRecommenderAction.Request>{ r });

System.debug('success  : ' + out[0].success);
System.debug('summary  : ' + out[0].summary);
for (CineWaveRecommendation rec : out[0].recommendations) {
    System.debug(rec.toSummary() +
                 '  retrieval=' + rec.alsScore +
                 '  ranked='    + rec.rankerScore +
                 '  source='    + rec.retrievalSource);
}

Map<String, Object> m = CineWaveGraphQLClient.getModelMetrics();
System.debug('model    : ' + m.get('modelVersion') + ' status=' + m.get('status'));
System.debug('NDCG@10  : ' + m.get('ndcgAt10') +
             ' (+' + m.get('ndcgLiftPctVsAls') + '% vs retrieval only)');
System.debug('caveats  : ' + m.get('caveats'));
