@IsTest
private class CineWaveControllerTest {

    @IsTest
    static void getRecommendationsReturnsSlate() {
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.SUCCESS));
        Test.startTest();
        List<CineWaveRecommendation> recs = CineWaveController.getRecommendations(1, 3);
        Test.stopTest();
        System.assertEquals(3, recs.size());
    }

    @IsTest
    static void nullKFallsBackToADefault() {
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.SUCCESS));
        Test.startTest();
        List<CineWaveRecommendation> recs = CineWaveController.getRecommendations(1, null);
        Test.stopTest();
        System.assertEquals(3, recs.size());
    }

    @IsTest
    static void metricsSurfaceToTheUi() {
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.METRICS));
        Test.startTest();
        Map<String, Object> m = CineWaveController.getModelMetrics();
        Test.stopTest();
        System.assertEquals('measured', m.get('status'));
    }

    @IsTest
    static void feedbackWritesThrough() {
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.FEEDBACK_OK));
        Test.startTest();
        System.assertEquals(true,
            CineWaveController.recordFeedback(1, 2858, 'play', 90));
        Test.stopTest();
    }

    @IsTest
    static void wrapsFailuresAsAuraHandledException() {
        Test.setMock(HttpCalloutMock.class, new CineWaveCalloutMock.Failing());
        Test.startTest();
        try {
            CineWaveController.getRecommendations(1, 3);
            System.assert(false, 'expected an AuraHandledException for the UI');
        } catch (AuraHandledException e) {
            System.assert(true);
        }
        Test.stopTest();
    }
}
