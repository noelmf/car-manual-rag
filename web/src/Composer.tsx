// Where the question is written.
//
// Three lines, fixed. It does not grow with the text: past three lines the
// field scrolls, so the send button never moves and the answer above it never
// gets pushed off the screen.

import { useState } from "react";
import type { ChangeEvent, KeyboardEvent, SyntheticEvent } from "react";

type ComposerProps = {
  onSend: (question: string) => void;
  busy?: boolean;
};

function ArrowUp() {
  return (
    <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
      <path
        d="M10 16V5.5M10 5l-4.6 4.6M10 5l4.6 4.6"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.1"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function Composer({ onSend, busy = false }: ComposerProps) {
  const [question, setQuestion] = useState("");
  const ready = question.trim() !== "" && !busy;

  // Submitting and pressing Enter arrive as different events, and the only
  // thing this needs from either is the chance to stop the default.
  function send(event: SyntheticEvent) {
    event.preventDefault();
    if (!ready) return;
    onSend(question.trim());
    // The empty field is the confirmation: the question went.
    setQuestion("");
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends. Shift+Enter is how a question that needs two lines stays
    // one question.
    if (event.key === "Enter" && !event.shiftKey) send(event);
  }

  return (
    <div>
      <form className="composer" onSubmit={send}>
        <label className="visually-hidden" htmlFor="question">
          Tu pregunta sobre el manual
        </label>
        <textarea
          id="question"
          rows={3}
          value={question}
          disabled={busy}
          placeholder="¿Cada cuánto se cambia el aceite?"
          onChange={(event: ChangeEvent<HTMLTextAreaElement>) => setQuestion(event.target.value)}
          onKeyDown={onKeyDown}
        />
        <div className="composer-foot">
          <button className="send" type="submit" disabled={!ready} aria-label="Enviar pregunta">
            <ArrowUp />
          </button>
        </div>
      </form>

      <p className="hint">
        <kbd>Enter</kbd> envía · <kbd>Shift</kbd>+<kbd>Enter</kbd> salta línea
      </p>
    </div>
  );
}
