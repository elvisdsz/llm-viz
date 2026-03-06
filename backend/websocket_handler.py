"""
WebSocket endpoint logic.

Each client connection gets its own handler coroutine.  The handler:
1. Receives a GenerateRequest JSON message from the client.
2. Runs generate_tokens() in a thread pool (it's CPU-bound).
3. For each GenerationStep, sends three messages: TokenEvent, AttentionData, ActivationData.
4. Sends GenerationComplete when done (or ErrorEvent on failure).

Cancellation: if the client disconnects mid-stream the WebSocket send will
raise an exception, which we catch and exit cleanly.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import WebSocket, WebSocketDisconnect

from backend.generator import generate_tokens
from backend.schemas import (
    ActivationData,
    AttentionData,
    ErrorEvent,
    GenerateRequest,
    GenerationComplete,
    TokenEvent,
)

# Thread pool for running the CPU-bound generation loop
_executor = ThreadPoolExecutor(max_workers=1)


async def handle_generation(websocket: WebSocket) -> None:
    """Main WebSocket handler. Call this from the FastAPI endpoint."""
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        data = json.loads(raw)
        request = GenerateRequest(**data)
    except Exception as exc:
        await websocket.send_text(
            ErrorEvent(message=f"Bad request: {exc}").model_dump_json()
        )
        await websocket.close()
        return

    loop = asyncio.get_running_loop()
    total = 0

    try:
        # Run generator in a thread so the event loop stays responsive.
        # We use run_in_executor with a wrapper that yields steps into a queue.
        queue: asyncio.Queue = asyncio.Queue()

        def _run_generator():
            try:
                for step in generate_tokens(
                    request.prompt,
                    request.max_new_tokens,
                    request.temperature,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, step)
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        loop.run_in_executor(_executor, _run_generator)

        errored = False
        while True:
            item = await queue.get()

            if item is None:
                # Sentinel: generation finished (or errored — handled below)
                break

            if isinstance(item, Exception):
                errored = True
                print(f"[generator error] {type(item).__name__}: {item}")
                await websocket.send_text(
                    ErrorEvent(message=f"{type(item).__name__}: {item}").model_dump_json()
                )
                break

            step = item
            total += 1

            token_msg = TokenEvent(
                token_str=step.token_str,
                token_id=step.token_id,
                token_index=step.token_index,
            )
            attn_msg = AttentionData(
                token_index=step.token_index,
                keys=step.attention_keys,
                weights=step.attention_weights,
            )
            act_msg = ActivationData(
                token_index=step.token_index,
                layer_count=step.layer_count,
                hidden_size=step.hidden_size,
                activations=step.activations,
            )

            await websocket.send_text(token_msg.model_dump_json())
            await websocket.send_text(attn_msg.model_dump_json())
            await websocket.send_text(act_msg.model_dump_json())

        # Only send completion if generation actually finished (not errored)
        if not errored:
            await websocket.send_text(
                GenerationComplete(total_tokens=total).model_dump_json()
            )

    except WebSocketDisconnect:
        # Client navigated away or clicked Stop — not an error
        print("Client disconnected during generation.")
    except Exception as exc:
        try:
            await websocket.send_text(
                ErrorEvent(message=str(exc)).model_dump_json()
            )
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
