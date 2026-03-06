/**
 * Thin WebSocket wrapper.
 *
 * Usage:
 *   const ws = new WSClient("ws://localhost:8000/ws/generate");
 *   ws.on("token", handler);
 *   ws.connect({ type: "generate", prompt: "...", max_new_tokens: 40 });
 *   ws.disconnect();
 */

export class WSClient {
  /** @type {WebSocket|null} */
  #socket = null;
  /** @type {Map<string, Function[]>} */
  #handlers = new Map();

  constructor(url) {
    this.url = url;
  }

  /**
   * Register a handler for a message type (or "*" for all messages).
   * @param {string} type
   * @param {(data: object) => void} fn
   */
  on(type, fn) {
    if (!this.#handlers.has(type)) this.#handlers.set(type, []);
    this.#handlers.get(type).push(fn);
    return this;
  }

  /** Remove all handlers for a given type, or all handlers if no type given. */
  off(type) {
    if (type) this.#handlers.delete(type);
    else this.#handlers.clear();
    return this;
  }

  /**
   * Open the connection and immediately send an initial message.
   * @param {object} initMessage  — will be JSON.stringify'd
   */
  connect(initMessage) {
    if (this.#socket) this.disconnect();

    this.#socket = new WebSocket(this.url);

    this.#socket.addEventListener("open", () => {
      this.#socket.send(JSON.stringify(initMessage));
    });

    this.#socket.addEventListener("message", (event) => {
      let data;
      try { data = JSON.parse(event.data); } catch { return; }

      const type = data.type ?? "_unknown";
      const fns = [...(this.#handlers.get(type) ?? []), ...(this.#handlers.get("*") ?? [])];
      for (const fn of fns) fn(data);
    });

    this.#socket.addEventListener("error", (event) => {
      const fns = this.#handlers.get("_error") ?? [];
      for (const fn of fns) fn(event);
    });

    this.#socket.addEventListener("close", (event) => {
      const fns = this.#handlers.get("_close") ?? [];
      for (const fn of fns) fn(event);
      this.#socket = null;
    });
  }

  /** Close the WebSocket gracefully. */
  disconnect() {
    if (this.#socket) {
      this.#socket.close(1000, "user stopped");
      this.#socket = null;
    }
  }

  get connected() {
    return this.#socket !== null && this.#socket.readyState === WebSocket.OPEN;
  }
}
