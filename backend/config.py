"""All project-wide constants. Edit this file to tune the model or visualization."""

MODEL_NAME = "openai-community/gpt2"

# Generation defaults
MAX_NEW_TOKENS = 50
DEFAULT_TEMPERATURE = 1.0

# Server
HOST = "0.0.0.0"
PORT = 8000

# GPT-2 Small architecture constants (used by visualizers before model is loaded)
NUM_LAYERS = 12   # transformer blocks
NUM_HEADS = 12    # attention heads per layer
HIDDEN_SIZE = 768

# Number of top neurons (by |activation|) to send per layer per step
TOP_N_NEURONS = 20

# WebSocket close codes
WS_NORMAL_CLOSE = 1000
