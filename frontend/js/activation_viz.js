/**
 * Neuron activation bar chart.
 *
 * Renders a horizontal bar chart of the top-N neurons for the selected layer.
 * Blue = positive activation, red = negative.
 *
 * Data is cached per token_index for replay when a past token is clicked.
 */

const BAR_H      = 18;   // height of each bar row (px)
const BAR_GAP    = 4;    // gap between rows
const LABEL_W    = 72;   // width reserved for neuron-index label
const VALUE_W    = 52;   // width reserved for value label on right
const MAX_BAR_W  = 220;  // max bar width in pixels

const POS_COLOR = "#60a5fa";   // blue
const NEG_COLOR = "#f87171";   // red
const TEXT_COLOR = "#8892a4";

export class ActivationViz {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {HTMLSelectElement} layerSelect
   */
  constructor(canvas, layerSelect) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.layerSelect = layerSelect;

    /** @type {Map<number, {layer_count, activations: Array<Array<[number,number]>>}>} */
    this.history = new Map();

    this.currentLayer = 0;

    this.layerSelect.addEventListener("change", () => {
      this.currentLayer = parseInt(this.layerSelect.value, 10);
      const latest = this._getLatestData();
      if (latest) this._render(latest);
    });
  }

  /** Populate layer dropdown. */
  initDropdown(layerCount) {
    this.layerSelect.innerHTML = "";
    for (let i = 0; i < layerCount; i++) {
      const opt = document.createElement("option");
      opt.value = i;
      opt.textContent = "Layer " + i;
      this.layerSelect.appendChild(opt);
    }
    this.currentLayer = 0;
  }

  /**
   * @param {{ token_index: number, layer_count: number, activations: Array<Array<[number,number]>> }} data
   */
  update(data) {
    this.history.set(data.token_index, data);
    this._render(data);
  }

  /** Replay visualization for a historical token. */
  replay(tokenIndex) {
    const data = this.history.get(tokenIndex);
    if (data) this._render(data);
  }

  /** Clear canvas + history. */
  reset() {
    this.history.clear();
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  // ── Private ──────────────────────────────────────────────────────

  _render(data) {
    const layer = Math.min(this.currentLayer, data.activations.length - 1);
    const neurons = data.activations[layer];   // [(idx, val), ...]

    if (!neurons || neurons.length === 0) return;

    const n       = neurons.length;
    const totalH  = n * (BAR_H + BAR_GAP) + 12;
    const totalW  = LABEL_W + MAX_BAR_W + VALUE_W + 16;

    this.canvas.width  = totalW;
    this.canvas.height = totalH;

    const ctx = this.ctx;
    ctx.clearRect(0, 0, totalW, totalH);

    // Find max |value| for scale
    const maxAbs = Math.max(...neurons.map(([, v]) => Math.abs(v)), 1e-6);

    ctx.font = `11px var(--mono, monospace)`;

    for (let i = 0; i < n; i++) {
      const [neuronIdx, value] = neurons[i];
      const y = i * (BAR_H + BAR_GAP) + 6;

      // Neuron index label
      ctx.fillStyle = TEXT_COLOR;
      ctx.textAlign = "right";
      ctx.fillText(`n${neuronIdx}`, LABEL_W - 6, y + BAR_H - 4);

      // Bar
      const barW = Math.abs(value) / maxAbs * MAX_BAR_W;
      ctx.fillStyle = value >= 0 ? POS_COLOR : NEG_COLOR;
      ctx.fillRect(LABEL_W, y, barW, BAR_H);

      // Value label
      ctx.fillStyle = TEXT_COLOR;
      ctx.textAlign = "left";
      ctx.fillText(value.toFixed(2), LABEL_W + MAX_BAR_W + 6, y + BAR_H - 4);
    }
  }

  _getLatestData() {
    if (this.history.size === 0) return null;
    const maxIdx = Math.max(...this.history.keys());
    return this.history.get(maxIdx) ?? null;
  }
}
