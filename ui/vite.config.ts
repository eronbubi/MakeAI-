import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The built UI is served by the MakeAI Python server (makeai/web).
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../makeai/web", emptyOutDir: true, chunkSizeWarningLimit: 900 },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:7860", "/ws": { target: "ws://127.0.0.1:7860", ws: true }, "/s/": "http://127.0.0.1:7860" },
  },
});
