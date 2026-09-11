/**
 * HTTP client for the CineWave GraphQL API.
 *
 * Every callout goes through the CineWave_API Named Credential, so the endpoint
 * and the secret live in metadata rather than in code or a custom setting.
 *
 * Why GraphQL rather than one REST call per use case: a Flow that only needs
 * three titles and a Lightning card that needs posters and both stage scores
 * are the same endpoint here, each asking for exactly the fields it renders.
 * Over a metered callout with a governor-limited timeout, not shipping fields
 * nobody reads is worth the extra query string.
 */
public with sharing class CineWaveGraphQLClient {

    @TestVisible private static final String ENDPOINT = 'callout:CineWave_API/graphql';
    @TestVisible private static final Integer TIMEOUT_MS = 20000;

    /** Fields the recommendation queries request. Kept in one place so the
     *  Flow action and the LWC controller cannot drift apart. */
    private static final String RECO_FIELDS =
        'itemId title primaryGenre year posterUrl score alsScore rankerScore ' +
        'retrievalSource explorationSlot';

    /**
     * Execute a GraphQL document and return its `data` map.
     *
     * @throws CineWaveException on transport failure, a non-200 status, or a
     *         populated `errors` array. GraphQL answers 200 with an errors
     *         array on a resolver failure, so checking the status code alone
     *         would let a failed query through as a success.
     */
    public static Map<String, Object> execute(String query, Map<String, Object> variables) {
        Map<String, Object> body = new Map<String, Object>{ 'query' => query };
        if (variables != null && !variables.isEmpty()) {
            body.put('variables', variables);
        }

        HttpRequest req = new HttpRequest();
        req.setEndpoint(ENDPOINT);
        req.setMethod('POST');
        req.setHeader('Content-Type', 'application/json');
        req.setHeader('Accept', 'application/json');
        req.setTimeout(TIMEOUT_MS);
        req.setBody(JSON.serialize(body));

        HttpResponse res;
        try {
            res = new Http().send(req);
        } catch (System.CalloutException e) {
            throw new CineWaveException('CineWave API unreachable: ' + e.getMessage(), e);
        }

        if (res.getStatusCode() != 200) {
            throw new CineWaveException(
                'CineWave API returned HTTP ' + res.getStatusCode() + ': ' + res.getStatus());
        }

        Map<String, Object> parsed;
        try {
            parsed = (Map<String, Object>) JSON.deserializeUntyped(res.getBody());
        } catch (Exception e) {
            throw new CineWaveException('CineWave API returned malformed JSON');
        }

        List<Object> errors = (List<Object>) parsed.get('errors');
        if (errors != null && !errors.isEmpty()) {
            Map<String, Object> first = (Map<String, Object>) errors[0];
            throw new CineWaveException('CineWave GraphQL error: ' + first.get('message'));
        }

        Map<String, Object> data = (Map<String, Object>) parsed.get('data');
        if (data == null) {
            throw new CineWaveException('CineWave API returned no data block');
        }
        return data;
    }

    /** Personalised slate for one user. */
    public static List<CineWaveRecommendation> getRecommendations(Integer userId, Integer k) {
        String query =
            'query Recs($userId: Int!, $k: Int!) {' +
            '  recommendations(userId: $userId, k: $k) {' +
            '    modelVersion latencyMs diversityScore' +
            '    items { ' + RECO_FIELDS + ' }' +
            '  }' +
            '}';

        Map<String, Object> data = execute(query, new Map<String, Object>{
            'userId' => userId, 'k' => k
        });

        Map<String, Object> slate = (Map<String, Object>) data.get('recommendations');
        List<CineWaveRecommendation> out = new List<CineWaveRecommendation>();
        if (slate == null) return out;

        for (Object node : (List<Object>) slate.get('items')) {
            out.add(CineWaveRecommendation.fromNode((Map<String, Object>) node));
        }
        return out;
    }

    /**
     * The model's measured offline metrics.
     *
     * `status` is 'measured' only when the API has a trained bundle loaded; it
     * reports 'unavailable' rather than returning defaults, so a Salesforce
     * consumer can tell the difference between a good model and no model.
     */
    public static Map<String, Object> getModelMetrics() {
        String query =
            'query { modelMetrics {' +
            '  status modelVersion ndcgAt10 ndcgAt10AlsOnly ndcgLiftPctVsAls' +
            '  ipsNdcgAt10 bestBaseline bestBaselineNdcg caveats' +
            '} }';
        return (Map<String, Object>) execute(query, null).get('modelMetrics');
    }

    /** Record an interaction back into the recommender's feedback loop. */
    public static Boolean recordFeedback(Integer userId, Integer itemId,
                                         String eventName, Decimal dwellSeconds) {
        String query =
            'mutation Fb($u: Int!, $i: Int!, $e: String!, $d: Float!) {' +
            '  recordFeedback(userId: $u, itemId: $i, event: $e, dwellSeconds: $d) {' +
            '    accepted message' +
            '  }' +
            '}';
        Map<String, Object> data = execute(query, new Map<String, Object>{
            'u' => userId, 'i' => itemId, 'e' => eventName,
            'd' => (dwellSeconds == null ? 0 : dwellSeconds)
        });
        Map<String, Object> ack = (Map<String, Object>) data.get('recordFeedback');
        return ack != null && Boolean.valueOf(ack.get('accepted'));
    }
}
