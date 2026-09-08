import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";

// index.html carries this element, so its absence is a broken build rather
// than something to paper over with a non-null assertion.
const root = document.getElementById("root");
if (!root) throw new Error("no #root in the page: the HTML and this script disagree");

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
