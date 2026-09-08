// Choosing the car, in one field instead of four dropdowns.
//
// Brand, model, year and edition land on exactly one manual, and that choice
// is what every later stage filters by. A driver knows their car as a phrase,
// not as a cascade, so they type what they remember in any order and the list
// keeps what still fits.

import { useMemo, useRef, useState } from "react";
import type { KeyboardEvent } from "react";

/** The picker as /api/catalog sends it: brand -> model -> year -> edition. */
export type Catalog = Record<string, Record<string, Record<string, Record<string, string>>>>;

export type Vehicle = {
  brand: string;
  model: string;
  year: string;
  edition: string;
  manualId: string;
};

type PickerProps = {
  vehicles: Vehicle[];
  onPick: (vehicle: Vehicle) => void;
};

/** How many rows are drawn at once. Past this nobody is reading, they type. */
export const SHOWN = 60;

/** Every path through the nested catalogue, newest year first. */
export function flatten(catalog: Catalog): Vehicle[] {
  const rows: Vehicle[] = [];
  for (const [brand, models] of Object.entries(catalog)) {
    for (const [model, years] of Object.entries(models)) {
      for (const [year, editions] of Object.entries(years)) {
        for (const [edition, manualId] of Object.entries(editions)) {
          rows.push({ brand, model, year, edition, manualId });
        }
      }
    }
  }
  return rows.sort(
    (a, b) =>
      a.brand.localeCompare(b.brand) ||
      a.model.localeCompare(b.model) ||
      Number(b.year) - Number(a.year) ||
      a.edition.localeCompare(b.edition),
  );
}

/** How a vehicle is written, in one place, so the list and the field agree. */
export function label(vehicle: Vehicle): string {
  return `${vehicle.brand} ${vehicle.model} ${vehicle.year} (${vehicle.edition})`;
}

/** The words to match on. Brackets are punctuation this field writes itself:
 *  left in, reopening the list after a choice would match nothing. */
export function terms(query: string): string[] {
  return query
    .toLowerCase()
    .replace(/[()]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

/** Every term has to appear somewhere, in any order: 'ibiza 2026' and
 *  '2026 ibiza' are the same question. */
export function search(vehicles: Vehicle[], query: string): Vehicle[] {
  const wanted = terms(query);
  if (!wanted.length) return vehicles;
  return vehicles.filter((v) => {
    const haystack = `${v.model} ${v.year} ${v.edition} ${v.brand}`.toLowerCase();
    return wanted.every((term) => haystack.includes(term));
  });
}

function escapeForRegExp(term: string): string {
  return term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** The label split so the typed part can be marked. Odd pieces are matches. */
function pieces(text: string, wanted: string[]): string[] {
  if (!wanted.length) return [text];
  return text.split(new RegExp(`(${wanted.map(escapeForRegExp).join("|")})`, "gi"));
}

function Magnifier() {
  return (
    <svg width="17" height="17" viewBox="0 0 20 20" aria-hidden="true">
      <circle cx="9" cy="9" r="6" fill="none" stroke="currentColor" strokeWidth="2" />
      <path d="M13.5 13.5 17 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export default function Picker({ vehicles, onPick }: PickerProps) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [picked, setPicked] = useState(false);
  const field = useRef<HTMLInputElement>(null);

  const found = useMemo(() => search(vehicles, query), [vehicles, query]);
  const drawn = found.slice(0, SHOWN);
  const wanted = terms(query);

  function choose(index: number) {
    const vehicle = drawn[index];
    if (!vehicle) return;
    setQuery(label(vehicle));
    setPicked(true);
    setOpen(false);
    onPick(vehicle);
  }

  function onKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (!open) return setOpen(true);
      if (!drawn.length) return;
      const step = event.key === "ArrowDown" ? 1 : -1;
      setActive((was) => (was + step + drawn.length) % drawn.length);
    } else if (event.key === "Enter") {
      event.preventDefault();
      choose(active);
    } else if (event.key === "Escape") {
      setOpen(false);
    }
  }

  return (
    <div className="picker">
      <label className="field-label" htmlFor="vehiculo">
        Tu vehículo
      </label>

      {/* Once a vehicle is chosen the field stops being a search box: the text
          is an answer, not a query, so it takes the accent and the magnifier
          goes. Editing it means the choice no longer stands. */}
      <div className={picked ? "box picked" : "box"}>
        {!picked && <Magnifier />}
        <input
          id="vehiculo"
          ref={field}
          type="text"
          role="combobox"
          autoComplete="off"
          aria-expanded={open}
          aria-controls="vehiculos"
          aria-autocomplete="list"
          aria-activedescendant={open && drawn.length ? `vehiculo-${active}` : undefined}
          placeholder="ibiza 2026"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setPicked(false);
            setActive(0);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setOpen(false)}
          onKeyDown={onKeyDown}
        />
        {query !== "" && (
          <button
            className="clear"
            type="button"
            aria-label="Borrar la búsqueda"
            onClick={() => {
              setQuery("");
              setPicked(false);
              setActive(0);
              setOpen(true);
              field.current?.focus();
            }}
          >
            ×
          </button>
        )}
      </div>

      <ul className="list" id="vehiculos" role="listbox" aria-label="Vehículos" hidden={!open}>
        {!found.length && <li className="empty">Ningún manual encaja con eso.</li>}

        {drawn.map((vehicle, index) => (
          <li
            key={vehicle.manualId + vehicle.year}
            id={`vehiculo-${index}`}
            className="option"
            role="option"
            aria-selected={index === active}
            // The field must keep the focus, or blur closes the list first.
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => choose(index)}
          >
            {pieces(label(vehicle), wanted).map((piece, i) =>
              i % 2 ? <mark key={i}>{piece}</mark> : piece,
            )}
          </li>
        ))}

        {found.length > SHOWN && (
          <li className="more">y {found.length - SHOWN} más — sigue escribiendo</li>
        )}
      </ul>
    </div>
  );
}
