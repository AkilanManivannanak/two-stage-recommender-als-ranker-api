/**
 * Typed representation of one CineWave recommendation.
 *
 * The GraphQL schema is the contract between the two systems: because the API
 * exposes a typed schema, this class maps onto it field-for-field instead of
 * hunting through an untyped JSON blob. `alsScore` and `rankerScore` are kept
 * separate on purpose — they are the two stages of the recommender, and
 * collapsing them into one number is what the upstream service used to do.
 */
public with sharing class CineWaveRecommendation {

    @AuraEnabled public Integer itemId          { get; set; }
    @AuraEnabled public String  title           { get; set; }
    @AuraEnabled public String  primaryGenre    { get; set; }
    @AuraEnabled public Integer year            { get; set; }
    @AuraEnabled public String  posterUrl       { get; set; }
    @AuraEnabled public Decimal score           { get; set; }
    @AuraEnabled public Decimal alsScore        { get; set; }
    @AuraEnabled public Decimal rankerScore     { get; set; }
    @AuraEnabled public String  retrievalSource { get; set; }
    @AuraEnabled public Boolean explorationSlot { get; set; }

    public CineWaveRecommendation() {}

    /** Build from one item node of the GraphQL `recommendations` payload. */
    public static CineWaveRecommendation fromNode(Map<String, Object> node) {
        CineWaveRecommendation r = new CineWaveRecommendation();
        r.itemId          = toInt(node.get('itemId'));
        r.title           = (String)  node.get('title');
        r.primaryGenre    = (String)  node.get('primaryGenre');
        r.year            = toInt(node.get('year'));
        r.posterUrl       = (String)  node.get('posterUrl');
        r.score           = toDec(node.get('score'));
        r.alsScore        = toDec(node.get('alsScore'));
        r.rankerScore     = toDec(node.get('rankerScore'));
        r.retrievalSource = (String)  node.get('retrievalSource');
        r.explorationSlot = Boolean.valueOf(node.get('explorationSlot'));
        return r;
    }

    /** One-line form for an Agentforce prompt or a Flow text template. */
    public String toSummary() {
        String yr = (year == null || year == 0) ? '' : ' (' + year + ')';
        return title + yr + ' — ' + primaryGenre;
    }

    private static Integer toInt(Object v) {
        if (v == null) return null;
        return Integer.valueOf(String.valueOf(v));
    }
    private static Decimal toDec(Object v) {
        if (v == null) return null;
        return Decimal.valueOf(String.valueOf(v));
    }
}
