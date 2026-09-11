@IsTest
private class CineWaveGraphQLClientTest {

    @IsTest
    static void returnsParsedRecommendations() {
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.SUCCESS));
        Test.startTest();
        List<CineWaveRecommendation> recs = CineWaveGraphQLClient.getRecommendations(1, 3);
        Test.stopTest();

        System.assertEquals(3, recs.size(), 'expected three recommendations');
        CineWaveRecommendation first = recs[0];
        System.assertEquals(2858, first.itemId);
        System.assertEquals('American Beauty', first.title);
        System.assertEquals('collaborative', first.retrievalSource);
        System.assertEquals(false, first.explorationSlot);
        // The two stages must remain distinguishable end to end.
        System.assertNotEquals(first.alsScore, first.rankerScore,
            'alsScore and rankerScore must stay distinct across the boundary');
        System.assertEquals('American Beauty (1999) — Comedy', first.toSummary());
    }

    @IsTest
    static void sendsAWellFormedGraphQLPost() {
        CineWaveCalloutMock mock =
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.SUCCESS);
        Test.setMock(HttpCalloutMock.class, mock);
        Test.startTest();
        CineWaveGraphQLClient.getRecommendations(42, 7);
        Test.stopTest();

        HttpRequest req = mock.capturedRequest;
        System.assertEquals('POST', req.getMethod());
        System.assert(req.getEndpoint().startsWith('callout:CineWave_API'),
            'must go through the Named Credential, not a hardcoded host');
        Map<String, Object> body =
            (Map<String, Object>) JSON.deserializeUntyped(req.getBody());
        System.assert(((String) body.get('query')).contains('recommendations'));
        Map<String, Object> vars = (Map<String, Object>) body.get('variables');
        System.assertEquals(42, vars.get('userId'));
        System.assertEquals(7, vars.get('k'));
    }

    @IsTest
    static void emptySlateReturnsEmptyListNotNull() {
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.EMPTY_SLATE));
        Test.startTest();
        List<CineWaveRecommendation> recs = CineWaveGraphQLClient.getRecommendations(1, 5);
        Test.stopTest();
        System.assertEquals(0, recs.size());
    }

    @IsTest
    static void parsesModelMetricsWithoutInventingDefaults() {
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.METRICS));
        Test.startTest();
        Map<String, Object> m = CineWaveGraphQLClient.getModelMetrics();
        Test.stopTest();
        System.assertEquals('measured', m.get('status'));
        System.assertEquals(0.0516, (Decimal) m.get('ndcgAt10'));
        System.assert(((List<Object>) m.get('caveats')).size() > 0,
            'metrics must carry their caveats across the boundary');
    }

    @IsTest
    static void recordsFeedback() {
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.FEEDBACK_OK));
        Test.startTest();
        Boolean ok = CineWaveGraphQLClient.recordFeedback(1, 2858, 'play', 120);
        Test.stopTest();
        System.assertEquals(true, ok);
    }

    @IsTest
    static void reportsRejectedFeedback() {
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.FEEDBACK_REJECTED));
        Test.startTest();
        Boolean ok = CineWaveGraphQLClient.recordFeedback(1, 2858, 'teleport', 0);
        Test.stopTest();
        System.assertEquals(false, ok);
    }

    @IsTest
    static void throwsOnHttpError() {
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.HTTP_500));
        Test.startTest();
        try {
            CineWaveGraphQLClient.getRecommendations(1, 3);
            System.assert(false, 'expected CineWaveException on HTTP 500');
        } catch (CineWaveException e) {
            System.assert(e.getMessage().contains('500'), e.getMessage());
        }
        Test.stopTest();
    }

    @IsTest
    static void throwsOnGraphQLErrorsDespiteHttp200() {
        // The branch that matters: GraphQL answers 200 with an errors array.
        // Checking only the status code would treat this as a success.
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.GRAPHQL_ERROR));
        Test.startTest();
        try {
            CineWaveGraphQLClient.getRecommendations(1, 3);
            System.assert(false, 'expected CineWaveException on a GraphQL errors array');
        } catch (CineWaveException e) {
            System.assert(e.getMessage().contains('GraphQL error'), e.getMessage());
        }
        Test.stopTest();
    }

    @IsTest
    static void throwsOnMalformedJson() {
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.MALFORMED));
        Test.startTest();
        try {
            CineWaveGraphQLClient.getRecommendations(1, 3);
            System.assert(false, 'expected CineWaveException on malformed JSON');
        } catch (CineWaveException e) {
            System.assert(e.getMessage().contains('malformed'), e.getMessage());
        }
        Test.stopTest();
    }

    @IsTest
    static void throwsWhenDataBlockMissing() {
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.NO_DATA));
        Test.startTest();
        try {
            CineWaveGraphQLClient.getModelMetrics();
            System.assert(false, 'expected CineWaveException when data is absent');
        } catch (CineWaveException e) {
            System.assert(e.getMessage().contains('no data'), e.getMessage());
        }
        Test.stopTest();
    }

    @IsTest
    static void throwsOnTransportFailure() {
        Test.setMock(HttpCalloutMock.class, new CineWaveCalloutMock.Failing());
        Test.startTest();
        try {
            CineWaveGraphQLClient.getRecommendations(1, 3);
            System.assert(false, 'expected CineWaveException on a transport failure');
        } catch (CineWaveException e) {
            System.assert(e.getMessage().contains('unreachable'), e.getMessage());
        }
        Test.stopTest();
    }
}
