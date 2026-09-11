@IsTest
private class CineWaveRecommendationTest {

    private static Map<String, Object> node(Object year) {
        return new Map<String, Object>{
            'itemId' => 2858, 'title' => 'American Beauty',
            'primaryGenre' => 'Comedy', 'year' => year,
            'posterUrl' => 'https://img/1.jpg', 'score' => 0.91,
            'alsScore' => 1.34, 'rankerScore' => 0.91,
            'retrievalSource' => 'collaborative', 'explorationSlot' => false
        };
    }

    @IsTest
    static void mapsEveryFieldFromTheGraphQLNode() {
        CineWaveRecommendation r = CineWaveRecommendation.fromNode(node(1999));
        System.assertEquals(2858, r.itemId);
        System.assertEquals('American Beauty', r.title);
        System.assertEquals('Comedy', r.primaryGenre);
        System.assertEquals(1999, r.year);
        System.assertEquals('https://img/1.jpg', r.posterUrl);
        System.assertEquals(0.91, r.score);
        System.assertEquals(1.34, r.alsScore);
        System.assertEquals(0.91, r.rankerScore);
        System.assertEquals('collaborative', r.retrievalSource);
        System.assertEquals(false, r.explorationSlot);
    }

    @IsTest
    static void keepsTheTwoStageScoresDistinct() {
        CineWaveRecommendation r = CineWaveRecommendation.fromNode(node(1999));
        System.assertNotEquals(r.alsScore, r.rankerScore,
            'retrieval and ranking scores must not collapse into one value');
    }

    @IsTest
    static void summaryIncludesYearWhenPresent() {
        System.assertEquals('American Beauty (1999) — Comedy',
            CineWaveRecommendation.fromNode(node(1999)).toSummary());
    }

    @IsTest
    static void summaryOmitsYearWhenNullOrZero() {
        System.assertEquals('American Beauty — Comedy',
            CineWaveRecommendation.fromNode(node(null)).toSummary());
        System.assertEquals('American Beauty — Comedy',
            CineWaveRecommendation.fromNode(node(0)).toSummary());
    }

    @IsTest
    static void handlesMissingFieldsWithoutThrowing() {
        CineWaveRecommendation r = CineWaveRecommendation.fromNode(
            new Map<String, Object>{ 'itemId' => 1, 'title' => 'Toy Story' });
        System.assertEquals(1, r.itemId);
        System.assertEquals('Toy Story', r.title);
        System.assertEquals(null, r.score);
        System.assertEquals(null, r.year);
        System.assertEquals(false, r.explorationSlot);
    }

    @IsTest
    static void parsesNumericsArrivingAsStrings() {
        // JSON.deserializeUntyped hands back Decimal or Integer depending on the
        // literal; the converters must accept either.
        CineWaveRecommendation r = CineWaveRecommendation.fromNode(
            new Map<String, Object>{
                'itemId' => '2858', 'title' => 'X', 'primaryGenre' => 'Drama',
                'year' => '1999', 'score' => '0.5', 'alsScore' => '1',
                'rankerScore' => '0.25' });
        System.assertEquals(2858, r.itemId);
        System.assertEquals(1999, r.year);
        System.assertEquals(0.5, r.score);
    }
}
