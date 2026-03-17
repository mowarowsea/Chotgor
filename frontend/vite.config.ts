import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/** バックエンド (FastAPI) へのプロキシ設定。 */
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 3000,
    proxy: {
      "/api": "http://localhost:8000",
      "/v1": "http://localhost:8000",
    },
  },
});
