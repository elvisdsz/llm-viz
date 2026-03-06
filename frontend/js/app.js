/**
 * app.js — entry point.
 *
 * Wires together all modules:
 *   WSClient → TokenDisplay + AttentionViz + ActivationViz
 *
 * Also manages UI state (enable/disable buttons, status bar).
 */

import { WSClient }       from "./websocket_client.js";
import { TokenDisplay }   from "./token_display.js";
import { AttentionViz }   from "./attention_viz.js";
import { ActivationViz }  from "./activation_viz.js";

// ── DOM refs ─────────────────────────────────────────────────────────────────
const promptInput    = document.getElementById("prompt-input");
const maxTokensInput = document.getElementById("max-tokens");
const temperatureInput = document.getElementById("temperature");
const generateBtn    = document.getElementById("generate-btn");
const stopBtn        = document.getElementById("stop-btn");
const statusBar      = document.getElementById("status-bar");
const tokenStream    = document.getElementById("token-stream");

const attnCanvas      = document.getElementById("attention-canvas");
const attnLayerSelect = document.getElementById("attn-layer-select");
const attnHeadSelect  = document.getElementById("attn-head-select");
const attnTooltip     = document.getElementById("attn-tooltip");

const actCanvas      = document.getElementById("activation-canvas");
const actLayerSelect = document.getElementById("act-layer-select");

// ── Module instances ─────────────────────────────────────────────────────────
const WS_URL = `ws://${location.host}/ws/generate`;
const wsClient = new WSClient(WS_URL);

/** @type {Map<number, {attention: object, activations: object}>} */
const stepHistory = new Map();

const tokenDisplay = new TokenDisplay(tokenStream, (tokenIndex) => {
  // User clicked a past token — replay its viz data
  const step = stepHistory.get(tokenIndex);
  if (!step) return;
  tokenDisplay.select(tokenIndex);
  if (step.attention)   attentionViz.replay(tokenIndex);
  if (step.activations) activationViz.replay(tokenIndex);
});

const attentionViz  = new AttentionViz(attnCanvas, attnLayerSelect, attnHeadSelect, attnTooltip);
const activationViz = new ActivationViz(actCanvas, actLayerSelect);

let attnDropdownReady = false;
let actDropdownReady  = false;

// ── UI helpers ────────────────────────────────────────────────────────────────
function setStatus(msg, cls = "") {
  statusBar.textContent = msg;
  statusBar.className = "status " + cls;
}

function setRunning(running) {
  generateBtn.disabled = running;
  stopBtn.disabled     = !running;
  promptInput.disabled = running;
}

// ── WebSocket message handlers ────────────────────────────────────────────────
wsClient.on("token", (data) => {
  tokenDisplay.addToken(data);
  tokenDisplay.select(data.token_index);

  // Ensure history entry exists
  if (!stepHistory.has(data.token_index)) stepHistory.set(data.token_index, {});
  stepHistory.get(data.token_index).token = data;

  setStatus(`Generating… token ${data.token_index + 1}`, "running");
});

wsClient.on("attention", (data) => {
  if (!attnDropdownReady) {
    attentionViz.initDropdowns(data.weights.length, data.weights[0]?.length ?? 12);
    attnDropdownReady = true;
  }

  if (!stepHistory.has(data.token_index)) stepHistory.set(data.token_index, {});
  stepHistory.get(data.token_index).attention = data;

  attentionViz.update(data);
});

wsClient.on("activations", (data) => {
  if (!actDropdownReady) {
    activationViz.initDropdown(data.layer_count);
    actDropdownReady = true;
  }

  if (!stepHistory.has(data.token_index)) stepHistory.set(data.token_index, {});
  stepHistory.get(data.token_index).activations = data;

  activationViz.update(data);
});

wsClient.on("complete", (data) => {
  setStatus(`Done. Generated ${data.total_tokens} tokens.`, "done");
  setRunning(false);
});

wsClient.on("error", (data) => {
  setStatus(`Error: ${data.message}`, "error");
  setRunning(false);
});

wsClient.on("_close", () => {
  // Only reset running state if we didn't already get a "complete" message
  if (generateBtn.disabled) {
    setRunning(false);
  }
});

wsClient.on("_error", () => {
  setStatus("WebSocket error — is the server running?", "error");
  setRunning(false);
});

// ── Button handlers ───────────────────────────────────────────────────────────
generateBtn.addEventListener("click", () => {
  const prompt = promptInput.value.trim();
  if (!prompt) { setStatus("Please enter a prompt.", "error"); return; }

  // Reset all viz state
  tokenDisplay.reset();
  attentionViz.reset();
  activationViz.reset();
  stepHistory.clear();
  attnDropdownReady = false;
  actDropdownReady  = false;

  setStatus("Connecting…", "running");
  setRunning(true);

  wsClient.connect({
    type: "generate",
    prompt,
    max_new_tokens: parseInt(maxTokensInput.value, 10) || 40,
    temperature: parseFloat(temperatureInput.value) || 1.0,
  });
});

stopBtn.addEventListener("click", () => {
  wsClient.disconnect();
  setStatus("Stopped.", "");
  setRunning(false);
});
