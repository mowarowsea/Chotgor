import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

/** バックエンド (FastAPI) へのプロキシ設定。 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  /** VITE_ALLOWED_HOSTS をカンマ区切りで複数指定可能。未設定時は空配列。 */
  const allowedHosts = env.VITE_ALLOWED_HOSTS
    ? env.VITE_ALLOWED_HOSTS.split(",").map((h) => h.trim())
    : [];
  return {
    plugins: [react()],
    server: {
      host: "0.0.0.0",
      port: 3000,
      allowedHosts,
      proxy: {
        "/api": "http://localhost:8000",
        "/v1": "http://localhost:8000",
        "/ui": "http://localhost:8000",
      },
    },
  };
});
