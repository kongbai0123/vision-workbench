/** Serialize image revisions without replacing edits made while a request is in flight. */
export class SaveQueue {
  constructor(persist, {onStatus = () => {}, onSaved = () => {}, delay = 800} = {}) {
    this.persist = persist;
    this.onStatus = onStatus;
    this.onSaved = onSaved;
    this.delay = delay;
    this.asset = null;
    this.generation = 0;
    this.savedGeneration = 0;
    this.inFlight = null;
    this.timer = null;
    this.error = null;
  }
  load(asset) {
    if (this.dirty || this.inFlight) throw Error('目前影像仍有尚未儲存的修改。');
    clearTimeout(this.timer);
    this.asset = asset;
    this.generation = 0;
    this.savedGeneration = 0;
    this.error = null;
    this.onStatus('saved');
  }
  get dirty() { return this.generation !== this.savedGeneration; }
  changed() {
    if (!this.asset) return;
    this.generation++;
    this.error = null;
    this.asset.review_state = 'pending';
    this.onStatus('dirty');
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.flush().catch(() => {}), this.delay);
  }
  async flush() {
    clearTimeout(this.timer);
    if (this.inFlight) return this.inFlight;
    if (!this.dirty || !this.asset) return;
    this.inFlight = this.drain();
    try { await this.inFlight; }
    finally { this.inFlight = null; }
  }
  async drain() {
    try {
      while (this.dirty) {
        const asset = this.asset;
        const generation = this.generation;
        const payload = {shapes: structuredClone(asset.shapes), revision: asset.revision};
        this.onStatus('saving');
        const saved = await this.persist(asset.id, payload);
        if (this.asset !== asset) throw Error('儲存期間的圖片已變更。');
        asset.revision = saved.revision;
        asset.shape_count = asset.shapes.length;
        asset.review_state = saved.review_state || 'pending';
        this.savedGeneration = generation;
        this.error = null;
        this.onSaved(asset, saved);
      }
      this.onStatus('saved');
    } catch (error) {
      this.error = error;
      this.onStatus('error', error);
      throw error;
    }
  }
}
