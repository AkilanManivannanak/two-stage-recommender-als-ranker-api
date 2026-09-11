/**
 * Invocable action exposing CineWave to Agentforce, Flow and Prompt Builder.
 *
 * An Agentforce topic can call this as a custom action ("what should I watch?"),
 * and a Flow can drop it onto a canvas. Both get the same two-stage recommender
 * that serves the web client.
 *
 * Bulkification note: Flow hands invocables a LIST of requests, and Apex allows
 * at most 100 callouts per transaction. This action makes ONE callout per
 * request row, so it caps the inbound list rather than silently failing the
 * 101st row halfway through a transaction. The upstream GraphQL API is
 * per-user, so there is no batch query to collapse these into — the honest
 * thing is to bound the input and say so.
 */
public with sharing class CineWaveRecommenderAction {

    @TestVisible private static final Integer MAX_ROWS_PER_TRANSACTION = 90;

    public class Request {
        @InvocableVariable(required=true
            label='CineWave User Id'
            description='Numeric CineWave user id to personalise for.')
        public Integer userId;

        @InvocableVariable(
            label='How many titles'
            description='Number of recommendations to return. Defaults to 5.')
        public Integer numberOfTitles;

        @InvocableVariable(
            label='Genre filter'
            description='Optional. Keep only titles in this primary genre.')
        public String genre;
    }

    public class Result {
        @InvocableVariable(label='Success')
        public Boolean success;

        @InvocableVariable(label='Titles'
            description='One line per recommendation, ready for a prompt or email template.')
        public List<String> titles;

        @InvocableVariable(label='Summary'
            description='Single sentence naming the recommendations, for an agent to speak.')
        public String summary;

        @InvocableVariable(label='Recommendations')
        public List<CineWaveRecommendation> recommendations;

        @InvocableVariable(label='Error message')
        public String errorMessage;
    }

    @InvocableMethod(
        label='Get CineWave Recommendations'
        description='Returns personalised movie recommendations from the CineWave two-stage recommender.'
        category='CineWave'
        callout=true)
    public static List<Result> getRecommendations(List<Request> requests) {
        if (requests != null && requests.size() > MAX_ROWS_PER_TRANSACTION) {
            throw new CineWaveException(
                'CineWaveRecommenderAction received ' + requests.size() + ' rows but makes ' +
                'one callout per row; Apex allows 100 per transaction. Batch to ' +
                MAX_ROWS_PER_TRANSACTION + ' or fewer.');
        }

        List<Result> results = new List<Result>();
        for (Request req : requests) {
            Result r = new Result();
            r.titles = new List<String>();
            r.recommendations = new List<CineWaveRecommendation>();
            try {
                Integer k = (req.numberOfTitles == null || req.numberOfTitles <= 0)
                            ? 5 : Math.min(req.numberOfTitles, 50);
                // Over-fetch when filtering, so a genre filter does not return
                // two titles because the unfiltered top-k happened to miss.
                Integer fetch = String.isBlank(req.genre) ? k : Math.min(k * 5, 50);

                List<CineWaveRecommendation> recs =
                    CineWaveGraphQLClient.getRecommendations(req.userId, fetch);

                for (CineWaveRecommendation rec : recs) {
                    if (String.isNotBlank(req.genre) &&
                        !req.genre.equalsIgnoreCase(rec.primaryGenre)) {
                        continue;
                    }
                    r.recommendations.add(rec);
                    r.titles.add(rec.toSummary());
                    if (r.recommendations.size() >= k) break;
                }

                r.success = true;
                r.summary = buildSummary(r.titles, req.genre);
            } catch (Exception e) {
                // Return the failure as data. An agent action that throws gives
                // the user a stack trace; one that returns success=false lets
                // the conversation degrade gracefully.
                r.success = false;
                r.errorMessage = e.getMessage();
                r.summary = 'Recommendations are unavailable right now.';
            }
            results.add(r);
        }
        return results;
    }

    private static String buildSummary(List<String> titles, String genre) {
        if (titles.isEmpty()) {
            return String.isBlank(genre)
                ? 'No recommendations available.'
                : 'No ' + genre + ' titles matched.';
        }
        String lead = String.isBlank(genre)
            ? 'Recommended for you: '
            : 'Recommended ' + genre + ' titles: ';
        return lead + String.join(titles, '; ') + '.';
    }
}
