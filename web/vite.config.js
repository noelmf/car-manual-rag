// Vite serves the UI in development and sends /api to crag-serve, so the
// browser talks to one origin and the Python side needs no CORS. `pnpm build`
// writes web/dist, which crag-serve then serves itself.
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
