/**
 * Token stream display.
 *
 * Renders token <span> elements one by one.
 * Clicking a token fires the onTokenSelect callback with the stored step data.
 *
 * Token spans are coloured by log-probability if provided (Phase 3).
 */

export class TokenDisplay {
  /**
   * @param {HTMLElement} container
   * @param {(index: number) => void} onTokenSelect  — called on span click
   */
  constructor(container, onTokenSelect) {
    this.container = container;
    this.onTokenSelect = onTokenSelect;
    /** @type {Array<{token_str, token_id, token_index}>} */
    this.tokens = [];
    this.selectedIndex = -1;
  }

  /** Clear all tokens from display and internal state. */
  reset() {
    this.container.innerHTML = "";
    this.tokens = [];
    this.selectedIndex = -1;
  }

  /**
   * Append a new token span.
   * @param {{ token_str: string, token_id: number, token_index: number }} tokenEvent
   */
  addToken(tokenEvent) {
    const { token_str, token_id, token_index } = tokenEvent;
    this.tokens.push(tokenEvent);

    const span = document.createElement("span");
    span.className = "token-span";
    span.dataset.index = token_index;
    // GPT-2 uses byte-pair encoding; leading "Ġ" = space.  Replace with real space.
    span.textContent = token_str.replace(/Ġ/g, " ").replace(/Ċ/g, "\n");
    span.title = `id=${token_id}  pos=${token_index}`;

    span.addEventListener("click", () => {
      this.select(token_index);
      this.onTokenSelect(token_index);
    });

    this.container.appendChild(span);
  }

  /**
   * Highlight the span at token_index and deselect the previous one.
   * @param {number} index
   */
  select(index) {
    if (this.selectedIndex >= 0) {
      const prev = this.container.querySelector(`[data-index="${this.selectedIndex}"]`);
      if (prev) prev.classList.remove("selected");
    }
    const next = this.container.querySelector(`[data-index="${index}"]`);
    if (next) next.classList.add("selected");
    this.selectedIndex = index;
  }
}
