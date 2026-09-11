/**
 * Reusable HttpCalloutMock for the CineWave GraphQL API.
 *
 * Apex cannot make callouts in a test context, so every path through
 * CineWaveGraphQLClient is exercised against canned responses. The mock covers
 * the failure modes as well as the happy path — a non-200, a GraphQL 200 with a
 * populated `errors` array, malformed JSON, and a thrown CalloutException —
 * because those are the branches that silently swallow problems if untested.
 */
@IsTest
public class CineWaveCalloutMock implements HttpCalloutMock {

    public enum Mode { SUCCESS, METRICS, FEEDBACK_OK, FEEDBACK_REJECTED,
                       HTTP_500, GRAPHQL_ERROR, MALFORMED, NO_DATA, EMPTY_SLATE }

    private Mode mode;
    public HttpRequest capturedRequest;

    public CineWaveCalloutMock(Mode m) { this.mode = m; }

    public static String slateBody() {
        return '{"data":{"recommendations":{' +
               '"modelVersion":"cinewave-als64-lgbm-20260910",' +
               '"latencyMs":12.4,"diversityScore":0.42,"items":[' +
               '{"itemId":2858,"title":"American Beauty","primaryGenre":"Comedy",' +
               '"year":1999,"posterUrl":"https://img/1.jpg","score":0.91,' +
               '"alsScore":1.34,"rankerScore":0.91,"retrievalSource":"collaborative",' +
               '"explorationSlot":false},' +
               '{"itemId":1196,"title":"Star Wars: Episode V","primaryGenre":"Action",' +
               '"year":1980,"posterUrl":"https://img/2.jpg","score":0.88,' +
               '"alsScore":1.21,"rankerScore":0.88,"retrievalSource":"collaborative",' +
               '"explorationSlot":false},' +
               '{"itemId":2028,"title":"Saving Private Ryan","primaryGenre":"Action",' +
               '"year":1998,"posterUrl":"https://img/3.jpg","score":0.80,' +
               '"alsScore":1.10,"rankerScore":0.80,"retrievalSource":"semantic",' +
               '"explorationSlot":true}]}}}';
    }

    public HTTPResponse respond(HttpRequest req) {
        capturedRequest = req;
        HttpResponse res = new HttpResponse();
        res.setHeader('Content-Type', 'application/json');
        res.setStatusCode(200);
        res.setStatus('OK');

        if (mode == Mode.SUCCESS) {
            res.setBody(slateBody());
        } else if (mode == Mode.EMPTY_SLATE) {
            res.setBody('{"data":{"recommendations":{"items":[]}}}');
        } else if (mode == Mode.METRICS) {
            res.setBody('{"data":{"modelMetrics":{"status":"measured",' +
                '"modelVersion":"cinewave-als64-lgbm-20260910","ndcgAt10":0.0516,' +
                '"ndcgAt10AlsOnly":0.0411,"ndcgLiftPctVsAls":25.5,' +
                '"ipsNdcgAt10":0.0216,"bestBaseline":"cooccurrence",' +
                '"bestBaselineNdcg":0.0481,' +
                '"caveats":["Offline evaluation only. No live A/B test has been run."]}}}');
        } else if (mode == Mode.FEEDBACK_OK) {
            res.setBody('{"data":{"recordFeedback":{"accepted":true,"message":"recorded"}}}');
        } else if (mode == Mode.FEEDBACK_REJECTED) {
            res.setBody('{"data":{"recordFeedback":{"accepted":false,' +
                        '"message":"event must be one of [...]"}}}');
        } else if (mode == Mode.HTTP_500) {
            res.setStatusCode(500);
            res.setStatus('Internal Server Error');
            res.setBody('{"detail":"boom"}');
        } else if (mode == Mode.GRAPHQL_ERROR) {
            res.setBody('{"errors":[{"message":"Cannot query field \'nope\'"}],"data":null}');
        } else if (mode == Mode.MALFORMED) {
            res.setBody('not json at all');
        } else if (mode == Mode.NO_DATA) {
            res.setBody('{"extensions":{}}');
        }
        return res;
    }

    /** Mock that throws, to cover the transport-failure branch. */
    public class Failing implements HttpCalloutMock {
        public HTTPResponse respond(HttpRequest req) {
            throw new CalloutException('connection timed out');
        }
    }
}
