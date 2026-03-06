/**
 * Attention heatmap visualizer.
 *
 * Renders a (seq_len × seq_len) heatmap on a <canvas> element.
 * The user picks layer + head via dropdown menus.
 * Each cell (row=query, col=key) is coloured by attention weight 0..1.
 *
 * Data is cached per token_index so clicking old tokens replays the viz.
 */

const CELL_SIZE = 24;          // pixels per heatmap cell
const FONT_SIZE = 10;          // label font size

// Color scale: 0 → near-black, 1 → bright accent blue
function weightToColor(w) {
  const v = Math.min(1, Math.max(0, w));
  // Interpolate from #0f1117 (bg) to #6c8ef7 (accent)
  const r = Math.round(15  + v * (108 - 15));
  const g = Math.round(17  + v * (142 - 17));
  const b = Math.round(23  + v * (247 - 23));
  return `rgb(${r},${g},${b})`;
}

export class AttentionViz {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {HTMLSelectElement} layerSelect
   * @param {HTMLSelectElement} headSelect
   * @param {HTMLElement}       tooltip
   */
  constructor(canvas, layerSelect, headSelect, tooltip) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.layerSelect = layerSelect;
    this.headSelect = headSelect;
    this.tooltip = tooltip;

    /** @type {Map<number, {keys: string[], weights: number[][][]}> } */
    this.history = new Map();   // token_index → AttentionData

    this.currentLayer = 0;
    this.currentHead  = 0;

    this._setupDropdownListeners();
    this._setupTooltip();
  }

  /**
   * Initialise layer + head dropdowns.
   * @param {number} numLayers
   * @param {number} numHeads
   */
  initDropdowns(numLayers, numHeads) {
    this._populateSelect(this.layerSelect, numLayers, "Layer ");
    this._populateSelect(this.headSelect,  numHeads,  "Head ");
    this.currentLayer = 0;
    this.currentHead  = 0;
  }

  /**
   * Store incoming attention data and re-render if it's for the latest token.
   * @param {{ token_index: number, keys: string[], weights: number[][][] }} data
   */
  update(data) {
    this.history.set(data.token_index, data);
    this._render(data);
  }

  /**
   * Replay visualization for a historical token.
   * @param {number} tokenIndex
   */
  replay(tokenIndex) {
    const data = this.history.get(tokenIndex);
    if (data) this._render(data);
  }

  /** Clear canvas and history. */
  reset() {
    this.history.clear();
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  // ── Private ──────────────────────────────────────────────────────

  _render(data) {
    const { keys, weights } = data;
    const layer = this.currentLayer;
    const head  = this.currentHead;

    if (!weights[layer] || !weights[layer][head]) return;

    const row = weights[layer][head];  // float[] of length seq_len (last query row)
    const seqLen = keys.length;

    // For the full NxN view we'd need weights[layer][head] to be seq×seq.
    // Currently the server sends only the last-position row (1 × seq_len).
    // Draw a single row heatmap (1 × seqLen).

    const LABEL_W = 40;
    const LABEL_H = 20;

    const totalW = LABEL_W + seqLen * CELL_SIZE;
    const totalH = LABEL_H + 1 * CELL_SIZE;   // one query row

    this.canvas.width  = totalW;
    this.canvas.height = totalH;

    const ctx = this.ctx;
    ctx.clearRect(0, 0, totalW, totalH);

    // Draw column labels (key positions)
    ctx.font = `${FONT_SIZE}px var(--mono, monospace)`;
    ctx.fillStyle = "#8892a4";
    ctx.textAlign = "center";
    for (let col = 0; col < seqLen; col++) {
      const x = LABEL_W + col * CELL_SIZE + CELL_SIZE / 2;
      ctx.fillText(keys[col], x, LABEL_H - 4);
    }

    // Row label
    ctx.textAlign = "right";
    ctx.fillText("q", LABEL_W - 4, LABEL_H + CELL_SIZE / 2 + FONT_SIZE / 2);

    // Draw cells
    for (let col = 0; col < seqLen; col++) {
      const x = LABEL_W + col * CELL_SIZE;
      const y = LABEL_H;
      ctx.fillStyle = weightToColor(row[col]);
      ctx.fillRect(x, y, CELL_SIZE - 1, CELL_SIZE - 1);
    }

    // Store for tooltip lookups
    this._lastData  = data;
    this._labelW    = LABEL_W;
    this._labelH    = LABEL_H;
  }

  _setupDropdownListeners() {
    this.layerSelect.addEventListener("change", () => {
      this.currentLayer = parseInt(this.layerSelect.value, 10);
      const latest = this._getLatestData();
      if (latest) this._render(latest);
    });
    this.headSelect.addEventListener("change", () => {
      this.currentHead = parseInt(this.headSelect.value, 10);
      const latest = this._getLatestData();
      if (latest) this._render(latest);
    });
  }

  _getLatestData() {
    if (this.history.size === 0) return null;
    const maxIdx = Math.max(...this.history.keys());
    return this.history.get(maxIdx) ?? null;
  }

  _setupTooltip() {
    this.canvas.addEventListener("mousemove", (e) => {
      if (!this._lastData) { this.tooltip.classList.add("hidden"); return; }

      const rect = this.canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      const col = Math.floor((mx - this._labelW) / CELL_SIZE);
      const row = Math.floor((my - this._labelH) / CELL_SIZE);

      const layer = this.currentLayer;
      const head  = this.currentHead;
      const weights = this._lastData.weights;
      const keys    = this._lastData.keys;

      if (col >= 0 && col < keys.length && row === 0 &&
          weights[layer] && weights[layer][head]) {
        const val = weights[layer][head][col];
        this.tooltip.textContent = `key=${keys[col]}  weight=${val.toFixed(4)}`;
        this.tooltip.style.left = `${e.clientX + 12}px`;
        this.tooltip.style.top  = `${e.clientY - 8}px`;
        this.tooltip.classList.remove("hidden");
      } else {
        this.tooltip.classList.add("hidden");
      }
    });

    this.canvas.addEventListener("mouseleave", () => {
      this.tooltip.classList.add("hidden");
    });
  }

  _populateSelect(sel, count, prefix) {
    sel.innerHTML = "";
    for (let i = 0; i < count; i++) {
      const opt = document.createElement("option");
      opt.value = i;
      opt.textContent = prefix + i;
      sel.appendChild(opt);
    }
  }
}
