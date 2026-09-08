// The picker is the filter the whole design rests on: brand, model, year and
// edition have to land on exactly one manual. These tests type what a driver
// types and check what comes back.

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import Picker, { SHOWN, flatten, label, search, terms } from "./Picker";
import type { Catalog, Vehicle } from "./Picker";

const CATALOGO: Catalog = {
  SEAT: {
    Ibiza: {
      "2026": { "07.25": "SEAT_Ibiza_07.25", "11.25": "SEAT_Ibiza_11.25" },
      "2013": { "07.12": "SEAT_Ibiza_07.12" },
    },
    Alhambra: {
      "2023": { "06.22": "SEAT_Alhambra_06.22" },
    },
  },
};

const COCHES = flatten(CATALOGO);

function open(vehicles: Vehicle[] = COCHES) {
  const onPick = vi.fn();
  const view = render(<Picker vehicles={vehicles} onPick={onPick} />);
  const field = screen.getByLabelText("Tu vehículo") as HTMLInputElement;
  fireEvent.focus(field); // the list only exists once the field is live
  return { onPick, field, view };
}

function type(field: HTMLInputElement, value: string) {
  fireEvent.change(field, { target: { value } });
}

const rows = () => screen.getAllByRole("option").map((li) => li.textContent);

describe("flatten", () => {
  it("yields one vehicle per path through the catalogue", () => {
    expect(COCHES).toHaveLength(4);
  });

  it("keeps the manual each path lands on", () => {
    const ibiza = COCHES.find((v) => v.year === "2026" && v.edition === "11.25");
    expect(ibiza?.manualId).toBe("SEAT_Ibiza_11.25");
  });

  it("puts the newest year first, which is where a driver looks", () => {
    const years = COCHES.filter((v) => v.model === "Ibiza").map((v) => v.year);
    expect(years).toEqual(["2026", "2026", "2013"]);
  });

  it("is empty for an empty catalogue rather than throwing", () => {
    expect(flatten({})).toEqual([]);
  });
});

describe("label", () => {
  it("writes the vehicle as one line with the edition in brackets", () => {
    expect(label(COCHES[0]!)).toBe("SEAT Alhambra 2023 (06.22)");
  });
});

describe("terms", () => {
  it("drops the brackets the field writes itself", () => {
    // Without this, reopening the list after a choice would match nothing.
    expect(terms("SEAT Ibiza 2026 (11.25)")).toEqual(["seat", "ibiza", "2026", "11.25"]);
  });

  it("treats an empty query as no terms at all", () => {
    expect(terms("   ")).toEqual([]);
  });
});

describe("search", () => {
  it("returns everything when nothing has been typed", () => {
    expect(search(COCHES, "")).toHaveLength(4);
  });

  it("matches on any field", () => {
    expect(search(COCHES, "alhambra")).toHaveLength(1);
    expect(search(COCHES, "2026")).toHaveLength(2);
    expect(search(COCHES, "07.12")).toHaveLength(1);
  });

  it("needs every term, so terms narrow instead of widen", () => {
    expect(search(COCHES, "ibiza 2026")).toHaveLength(2);
    expect(search(COCHES, "ibiza 2026 11.25")).toHaveLength(1);
  });

  it("does not care about the order they were typed", () => {
    expect(search(COCHES, "2026 ibiza")).toEqual(search(COCHES, "ibiza 2026"));
  });

  it("finds nothing for a car that is not there", () => {
    expect(search(COCHES, "leon")).toEqual([]);
  });

  it("still matches once the field has written its own brackets", () => {
    expect(search(COCHES, "SEAT Ibiza 2026 (11.25)")).toHaveLength(1);
  });
});

describe("the list", () => {
  it("is closed until the field is focused", () => {
    const onPick = vi.fn();
    render(<Picker vehicles={COCHES} onPick={onPick} />);
    expect(screen.queryAllByRole("option")).toHaveLength(0);
  });

  it("shows every vehicle as one line", () => {
    open();
    expect(rows()).toContain("SEAT Ibiza 2026 (11.25)");
  });

  it("narrows as the question gets more precise", () => {
    const { field } = open();
    type(field, "ibiza 2026");
    expect(rows()).toEqual(["SEAT Ibiza 2026 (07.25)", "SEAT Ibiza 2026 (11.25)"]);
  });

  it("says so plainly when nothing matches", () => {
    const { field } = open();
    type(field, "leon");
    expect(screen.queryAllByRole("option")).toHaveLength(0);
    expect(screen.getByText("Ningún manual encaja con eso.")).toBeDefined();
  });

  it("marks what was typed inside each row", () => {
    const { view, field } = open();
    type(field, "ibiza");
    const marks = view.container.querySelectorAll("mark");
    expect([...marks].map((m) => m.textContent)).toContain("Ibiza");
  });

  it("draws only the first rows and counts the rest", () => {
    // With everything listed nobody reads; they type.
    const muchos: Vehicle[] = Array.from({ length: SHOWN + 5 }, (_, i) => ({
      brand: "SEAT",
      model: "Ibiza",
      year: String(2000 + i),
      edition: "01.00",
      manualId: "SEAT_Ibiza_" + i,
    }));
    open(muchos);
    expect(screen.getAllByRole("option")).toHaveLength(SHOWN);
    expect(screen.getByText("y 5 más — sigue escribiendo")).toBeDefined();
  });
});

describe("the keyboard", () => {
  it("moves down the list", () => {
    const { field } = open();
    fireEvent.keyDown(field, { key: "ArrowDown" });
    expect(screen.getAllByRole("option")[1]?.getAttribute("aria-selected")).toBe("true");
  });

  it("wraps round from the top", () => {
    const { field } = open();
    fireEvent.keyDown(field, { key: "ArrowUp" });
    const options = screen.getAllByRole("option");
    expect(options[options.length - 1]?.getAttribute("aria-selected")).toBe("true");
  });

  it("chooses with Enter", () => {
    const { field, onPick } = open();
    type(field, "ibiza 2026 11.25");
    fireEvent.keyDown(field, { key: "Enter" });
    expect(onPick).toHaveBeenCalledWith(
      expect.objectContaining({ manualId: "SEAT_Ibiza_11.25" }),
    );
  });

  it("chooses nothing when nothing matches", () => {
    const { field, onPick } = open();
    type(field, "leon");
    fireEvent.keyDown(field, { key: "Enter" });
    expect(onPick).not.toHaveBeenCalled();
  });

  it("reopens a closed list instead of moving inside it", () => {
    const { field } = open();
    fireEvent.keyDown(field, { key: "Escape" });
    fireEvent.keyDown(field, { key: "ArrowDown" });
    expect(screen.getAllByRole("option")[0]?.getAttribute("aria-selected")).toBe("true");
  });

  it("has nowhere to move when nothing matches", () => {
    const { field, onPick } = open();
    type(field, "leon");
    fireEvent.keyDown(field, { key: "ArrowDown" });
    fireEvent.keyDown(field, { key: "Enter" });
    expect(onPick).not.toHaveBeenCalled();
  });

  it("closes on Escape", () => {
    const { field } = open();
    fireEvent.keyDown(field, { key: "Escape" });
    expect(screen.queryAllByRole("option")).toHaveLength(0);
  });
});

describe("leaving the field", () => {
  it("closes the list, so it does not hang over the rest of the page", () => {
    const { field } = open();
    expect(screen.getAllByRole("option").length).toBeGreaterThan(0);
    fireEvent.blur(field);
    expect(screen.queryAllByRole("option")).toHaveLength(0);
  });

  it("does not blur when a row is pressed, or the row would vanish first", () => {
    // mousedown on the list has its default stopped for exactly this reason.
    const { view } = open();
    const row = screen.getByText("SEAT Alhambra 2023 (06.22)");
    const stopped = !fireEvent.mouseDown(row);
    expect(stopped).toBe(true);
    expect(view.container.querySelector(".list")).not.toBeNull();
  });
});

describe("choosing a vehicle", () => {
  it("hands the whole vehicle over, manual included", () => {
    const { onPick } = open();
    fireEvent.click(screen.getByText("SEAT Alhambra 2023 (06.22)"));
    expect(onPick).toHaveBeenCalledWith({
      brand: "SEAT",
      model: "Alhambra",
      year: "2023",
      edition: "06.22",
      manualId: "SEAT_Alhambra_06.22",
    });
  });

  it("writes the choice into the field", () => {
    const { field } = open();
    fireEvent.click(screen.getByText("SEAT Alhambra 2023 (06.22)"));
    expect(field.value).toBe("SEAT Alhambra 2023 (06.22)");
  });

  it("turns the field from a search box into an answer", () => {
    const { view } = open();
    expect(view.container.querySelector("svg")).not.toBeNull();
    fireEvent.click(screen.getByText("SEAT Alhambra 2023 (06.22)"));
    expect(view.container.querySelector(".box.picked")).not.toBeNull();
    expect(view.container.querySelector("svg")).toBeNull();
  });

  it("goes back to being a search box as soon as the text is edited", () => {
    const { view, field } = open();
    fireEvent.click(screen.getByText("SEAT Alhambra 2023 (06.22)"));
    type(field, "SEAT Alh");
    expect(view.container.querySelector(".box.picked")).toBeNull();
    expect(view.container.querySelector("svg")).not.toBeNull();
  });
});

describe("clearing", () => {
  it("has no button to press while the field is empty", () => {
    open();
    expect(screen.queryByLabelText("Borrar la búsqueda")).toBeNull();
  });

  it("empties the field and shows everything again", () => {
    const { field } = open();
    type(field, "alhambra");
    fireEvent.click(screen.getByLabelText("Borrar la búsqueda"));
    expect(field.value).toBe("");
    expect(screen.getAllByRole("option")).toHaveLength(4);
  });
});
