// One config for both jobs. Vitest reads this file, so the tests resolve
// imports exactly as the app does and there is no second build to keep in
// step with the first.
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: {
    // Vite serves the UI in development and sends /api to crag-serve, so the
    // browser talks to one origin and the Python side needs no CORS.
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  test: {
    // A component that reads the DOM needs a DOM to read.
    environment: "jsdom",
    setupFiles: ["./src/test-setup.ts"],
    coverage: {
      provider: "v8",
      // lcov is what Codecov reads; text is what a person reads locally.
      reporter: ["text", "lcov"],
      // Everything under src counts, whether a test imports it or not: a file
      // nobody tests should show as untested rather than as absent.
      include: ["src/**/*.{ts,tsx}"],
      exclude: [
        "src/**/*.test.tsx",
        "src/test-setup.ts",
        "src/vite-env.d.ts",
        // The entry point, left out for the same reason the Python side marks
        // its main() no cover: it wires things up and there is nothing in it
        // a test could assert that the wiring itself does not already prove.
        "src/main.tsx",
      ],
    },
  },
});
