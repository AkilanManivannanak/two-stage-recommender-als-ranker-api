import { LightningElement, api, wire } from 'lwc';
import { ShowToastEvent } from 'lightning/platformShowToastEvent';
import getRecommendations from '@salesforce/apex/CineWaveController.getRecommendations';
import getModelMetrics from '@salesforce/apex/CineWaveController.getModelMetrics';
import recordFeedback from '@salesforce/apex/CineWaveController.recordFeedback';

export default class CineWaveRecommendations extends LightningElement {
    @api cineWaveUserId = 1;
    @api numberOfTitles = 6;

    recommendations = [];
    metrics;
    error;
    loading = true;

    @wire(getRecommendations, { userId: '$cineWaveUserId', k: '$numberOfTitles' })
    wiredRecs({ data, error }) {
        this.loading = false;
        if (data) {
            this.recommendations = data.map((r) => ({
                ...r,
                key: r.itemId,
                subtitle: `${r.primaryGenre}${r.year ? ' · ' + r.year : ''}`,
                // Both stage scores are shown. Collapsing them into one number
                // is what hid a broken reranker upstream for months.
                scoreLabel: `retrieval ${r.alsScore?.toFixed(3)} → ranked ${r.rankerScore?.toFixed(3)}`
            }));
            this.error = undefined;
        } else if (error) {
            this.error = error.body?.message ?? 'Could not reach CineWave.';
            this.recommendations = [];
        }
    }

    @wire(getModelMetrics)
    wiredMetrics({ data }) {
        if (data) this.metrics = data;
    }

    get hasRecommendations() {
        return this.recommendations.length > 0;
    }

    /** Shown verbatim so nobody reads an offline number as a live A/B result. */
    get metricsLabel() {
        if (!this.metrics) return '';
        if (this.metrics.status !== 'measured') return 'Model metrics unavailable';
        return `NDCG@10 ${this.metrics.ndcgAt10} (offline, +${this.metrics.ndcgLiftPctVsAls}% vs retrieval only)`;
    }

    async handlePlay(event) {
        const itemId = parseInt(event.target.dataset.itemId, 10);
        try {
            await recordFeedback({
                userId: this.cineWaveUserId,
                itemId,
                eventName: 'play',
                dwellSeconds: 0
            });
            this.dispatchEvent(new ShowToastEvent({
                title: 'Saved', message: 'Sent to the recommender.', variant: 'success'
            }));
        } catch (e) {
            this.dispatchEvent(new ShowToastEvent({
                title: 'Not saved',
                message: e.body?.message ?? 'Feedback did not reach CineWave.',
                variant: 'error'
            }));
        }
    }
}
