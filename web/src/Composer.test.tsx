// The composer is driven by hand more than it is read, so the tests press the
// keys a person presses rather than reaching into the component's state.

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import Composer from "./Composer";

function open(props: Partial<Parameters<typeof Composer>[0]> = {}) {
  const onSend = vi.fn();
  render(<Composer onSend={onSend} {...props} />);
  return {
    onSend,
    field: screen.getByLabelText("Tu pregunta sobre el manual") as HTMLTextAreaElement,
    button: screen.getByLabelText("Enviar pregunta") as HTMLButtonElement,
  };
}

function type(field: HTMLTextAreaElement, value: string) {
  fireEvent.change(field, { target: { value } });
}

describe("the field", () => {
  it("is three lines tall", () => {
    // The whole point of the shape: it never grows, so the send button never
    // moves and the answer above it never gets pushed off the screen.
    expect(open().field.rows).toBe(3);
  });

  it("starts empty whatever the placeholder suggests", () => {
    const { field } = open();
    expect(field.value).toBe("");
    expect(field.placeholder).toBe("¿Cada cuánto se cambia el aceite?");
  });
});

describe("the send button", () => {
  it("cannot be pressed with nothing written", () => {
    expect(open().button.disabled).toBe(true);
  });

  it("wakes up once there is a question", () => {
    const { field, button } = open();
    type(field, "¿Cada cuánto se cambia el aceite?");
    expect(button.disabled).toBe(false);
  });

  it("stays asleep for whitespace, which is not a question", () => {
    const { field, button } = open();
    type(field, "   \n  ");
    expect(button.disabled).toBe(true);
  });

  it("sends what was written, trimmed", () => {
    const { field, button, onSend } = open();
    type(field, "  ¿Dónde está el gato?  ");
    fireEvent.click(button);
    expect(onSend).toHaveBeenCalledWith("¿Dónde está el gato?");
  });

  it("empties the field, which is the only confirmation there is", () => {
    const { field, button } = open();
    type(field, "¿Cómo se cambia una rueda?");
    fireEvent.click(button);
    expect(field.value).toBe("");
  });
});

describe("the keyboard", () => {
  it("sends on Enter", () => {
    const { field, onSend } = open();
    type(field, "¿Qué presión llevan las ruedas?");
    fireEvent.keyDown(field, { key: "Enter" });
    expect(onSend).toHaveBeenCalledWith("¿Qué presión llevan las ruedas?");
  });

  it("does not send on Shift+Enter, so a two-line question stays one question", () => {
    const { field, onSend } = open();
    type(field, "¿Qué presión llevan las ruedas");
    fireEvent.keyDown(field, { key: "Enter", shiftKey: true });
    expect(onSend).not.toHaveBeenCalled();
  });

  it("ignores Enter on an empty field", () => {
    const { field, onSend } = open();
    fireEvent.keyDown(field, { key: "Enter" });
    expect(onSend).not.toHaveBeenCalled();
  });

  it("leaves other keys alone", () => {
    const { field, onSend } = open();
    type(field, "¿Y el aceite?");
    fireEvent.keyDown(field, { key: "a" });
    expect(onSend).not.toHaveBeenCalled();
  });
});

describe("while an answer is being written", () => {
  it("takes no more questions", () => {
    const { field, button } = open({ busy: true });
    expect(field.disabled).toBe(true);
    expect(button.disabled).toBe(true);
  });

  it("will not send even one already typed", () => {
    const onSend = vi.fn();
    const { rerender } = render(<Composer onSend={onSend} />);
    const field = screen.getByLabelText("Tu pregunta sobre el manual");
    type(field as HTMLTextAreaElement, "¿Y esta?");
    rerender(<Composer onSend={onSend} busy />);
    fireEvent.keyDown(field, { key: "Enter" });
    expect(onSend).not.toHaveBeenCalled();
  });
});
