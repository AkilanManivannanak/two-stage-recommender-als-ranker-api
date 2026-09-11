@IsTest
private class CineWaveRecommenderActionTest {

    private static CineWaveRecommenderAction.Request req(Integer userId, Integer k, String genre) {
        CineWaveRecommenderAction.Request r = new CineWaveRecommenderAction.Request();
        r.userId = userId; r.numberOfTitles = k; r.genre = genre;
        return r;
    }

    @IsTest
    static void returnsTitlesAndSpeakableSummary() {
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.SUCCESS));
        Test.startTest();
        List<CineWaveRecommenderAction.Result> out =
            CineWaveRecommenderAction.getRecommendations(
                new List<CineWaveRecommenderAction.Request>{ req(1, 3, null) });
        Test.stopTest();

        System.assertEquals(1, out.size());
        CineWaveRecommenderAction.Result r = out[0];
        System.assertEquals(true, r.success);
        System.assertEquals(3, r.titles.size());
        System.assertEquals(3, r.recommendations.size());
        System.assert(r.summary.startsWith('Recommended for you: '), r.summary);
        System.assert(r.summary.contains('American Beauty (1999) — Comedy'), r.summary);
        System.assert(r.summary.endsWith('.'), r.summary);
    }

    @IsTest
    static void filtersByGenre() {
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.SUCCESS));
        Test.startTest();
        List<CineWaveRecommenderAction.Result> out =
            CineWaveRecommenderAction.getRecommendations(
                new List<CineWaveRecommenderAction.Request>{ req(1, 5, 'Action') });
        Test.stopTest();

        CineWaveRecommenderAction.Result r = out[0];
        System.assertEquals(true, r.success);
        System.assertEquals(2, r.recommendations.size(),
            'the canned slate holds exactly two Action titles');
        for (CineWaveRecommendation rec : r.recommendations) {
            System.assertEquals('Action', rec.primaryGenre);
        }
        System.assert(r.summary.startsWith('Recommended Action titles: '), r.summary);
    }

    @IsTest
    static void genreFilterIsCaseInsensitive() {
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.SUCCESS));
        Test.startTest();
        List<CineWaveRecommenderAction.Result> out =
            CineWaveRecommenderAction.getRecommendations(
                new List<CineWaveRecommenderAction.Request>{ req(1, 5, 'aCtIoN') });
        Test.stopTest();
        System.assertEquals(2, out[0].recommendations.size());
    }

    @IsTest
    static void respectsRequestedCount() {
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.SUCCESS));
        Test.startTest();
        List<CineWaveRecommenderAction.Result> out =
            CineWaveRecommenderAction.getRecommendations(
                new List<CineWaveRecommenderAction.Request>{ req(1, 2, null) });
        Test.stopTest();
        System.assertEquals(2, out[0].recommendations.size());
    }

    @IsTest
    static void defaultsToFiveWhenCountOmitted() {
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.SUCCESS));
        Test.startTest();
        List<CineWaveRecommenderAction.Result> out =
            CineWaveRecommenderAction.getRecommendations(
                new List<CineWaveRecommenderAction.Request>{ req(1, null, null) });
        Test.stopTest();
        // The mock returns three; the point is that a null count did not throw
        // and did not return zero.
        System.assertEquals(true, out[0].success);
        System.assertEquals(3, out[0].recommendations.size());
    }

    @IsTest
    static void degradesGracefullyWhenApiIsDown() {
        // An agent action that throws shows the user a stack trace. This one
        // must return success=false and a sentence the agent can say.
        Test.setMock(HttpCalloutMock.class, new CineWaveCalloutMock.Failing());
        Test.startTest();
        List<CineWaveRecommenderAction.Result> out =
            CineWaveRecommenderAction.getRecommendations(
                new List<CineWaveRecommenderAction.Request>{ req(1, 3, null) });
        Test.stopTest();

        CineWaveRecommenderAction.Result r = out[0];
        System.assertEquals(false, r.success);
        System.assertEquals(0, r.recommendations.size());
        System.assert(String.isNotBlank(r.errorMessage));
        System.assertEquals('Recommendations are unavailable right now.', r.summary);
    }

    @IsTest
    static void reportsNoMatchWhenGenreFilterEmptiesTheSlate() {
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.SUCCESS));
        Test.startTest();
        List<CineWaveRecommenderAction.Result> out =
            CineWaveRecommenderAction.getRecommendations(
                new List<CineWaveRecommenderAction.Request>{ req(1, 5, 'Documentary') });
        Test.stopTest();
        System.assertEquals(0, out[0].recommendations.size());
        System.assertEquals('No Documentary titles matched.', out[0].summary);
    }

    @IsTest
    static void rejectsOversizedBatchRatherThanBlowingCalloutLimits() {
        List<CineWaveRecommenderAction.Request> many =
            new List<CineWaveRecommenderAction.Request>();
        for (Integer i = 0; i < 101; i++) many.add(req(i + 1, 3, null));
        Test.setMock(HttpCalloutMock.class,
            new CineWaveCalloutMock(CineWaveCalloutMock.Mode.SUCCESS));
        Test.startTest();
        try {
            CineWaveRecommenderAction.getRecommendations(many);
            System.assert(false, 'expected the action to refuse an oversized batch');
        } catch (CineWaveException e) {
            System.assert(e.getMessage().contains('one callout per row'), e.getMessage());
        }
        Test.stopTest();
    }
}
