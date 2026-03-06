# LLM Visualizer

A web app that runs GPT-2 Small on-device and visualizes its internals — attention weights and neuron activations — in real time as it generates text.

Built for learning: the code is deliberately readable end-to-end.

---

## Architecture

```
Browser (Vanilla JS + Canvas)
        |
        |  WebSocket  (ws://localhost:8000/ws/generate)
        |
FastAPI + Uvicorn  (Python)
        |
HuggingFace Transformers  (GPT-2 Small, 124M params)
        |
PyTorch (CPU or CUDA)
```

### Data flow per token

```
User clicks Generate
  → WSClient sends GenerateRequest
  → websocket_handler receives it, spawns generator in thread pool
  → generator.py: forward pass with output_attentions=True, output_hidden_states=True
  → activation_extractor extracts top-20 neurons + attention row for last position
  → handler sends three messages: TokenEvent, AttentionData, ActivationData
  → JS dispatches by msg.type → tokenDisplay / attentionViz / activationViz
  → Canvas redraws
```

### Why use_cache=False?

Disabling the KV cache forces a full forward pass every step, producing a complete `(seq_len × seq_len)` attention matrix per layer/head. Slower (quadratic in sequence length), but the full matrix is far simpler to visualize than the partial incremental version. Fine for sequences under ~100 tokens on a laptop.

---

## Directory structure

```
llm-viz/
├── backend/
│   ├── config.py              # All constants
│   ├── model_loader.py        # Singleton GPT-2 loader
│   ├── activation_extractor.py # Tensor → serializable list helpers
│   ├── generator.py           # Autoregressive loop, yields GenerationStep
│   ├── websocket_handler.py   # WS endpoint, drives generator, streams messages
│   ├── schemas.py             # Pydantic message types
│   └── main.py                # FastAPI app, static file serving, lifespan
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── app.js             # Entry point, wires everything
│       ├── websocket_client.js
│       ├── token_display.js
│       ├── attention_viz.js   # Canvas heatmap
│       └── activation_viz.js  # Canvas bar chart
└── requirements.txt
```

---

## Quick start

### 1. Install dependencies

```bash
# Install PyTorch (with CUDA 12.6 — adjust the URL for your CUDA version)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies from pyproject.toml
uv sync
```

### 2. Start the server

```bash
uv run uvicorn backend.main:app --reload --port 8000
```

The first run downloads GPT-2 Small (~500 MB) from Hugging Face. You'll see:

```
Loading openai-community/gpt2 on cuda...
Model loaded.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Open the app

Navigate to [http://localhost:8000/app](http://localhost:8000/app)

---

## What the panels show

### Token Stream
Each generated token appears as a clickable span. Click any token to replay its attention and activation data in the visualization panels below.

### Attention Heatmap
Shows the attention weight distribution of the **last generated token** (query) over all preceding tokens (keys) for the selected layer and head.

- Brighter cell = higher attention weight
- Use the **Layer** and **Head** dropdowns to explore all 12×12 = 144 heads
- Hover over cells to see exact weight values

### Neuron Activations
Shows the top-20 neurons (by absolute activation magnitude) for the selected layer at the last generated token's position.

- **Blue bars** = positive activation
- **Red bars** = negative activation
- Use the **Layer** dropdown to explore all 13 layers (embedding + 12 transformer blocks)

---

## Configuration

Edit `backend/config.py` to change:

| Constant | Default | Description |
|---|---|---|
| `MODEL_NAME` | `openai-community/gpt2` | HuggingFace model ID |
| `TOP_N_NEURONS` | `20` | Neurons shown per layer |
| `MAX_NEW_TOKENS` | `50` | Default generation length |
| `PORT` | `8000` | Server port |

To use GPT-2 Medium (24 layers, 16 heads, 1024 hidden): change `MODEL_NAME` to `openai-community/gpt2-medium` and update `NUM_LAYERS`, `NUM_HEADS`, `HIDDEN_SIZE`.

---

## WebSocket message schema

```
GenerateRequest   { type:"generate", prompt, max_new_tokens, temperature }
TokenEvent        { type:"token", token_str, token_id, token_index }
AttentionData     { type:"attention", token_index, keys:[str], weights:[[[float]]] }
                    weights[layer][head][key_position] ∈ [0,1]
ActivationData    { type:"activations", token_index, layer_count, hidden_size,
                    activations:[[(neuron_idx, value)]] }
GenerationComplete { type:"complete", total_tokens }
ErrorEvent         { type:"error", message }
```
