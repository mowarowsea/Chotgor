import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

/** バックエンド (FastAPI) へのプロキシ設定。 */
export default defineConfig(({ mode, command }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  /** VITE_ALLOWED_HOSTS をカンマ区切りで複数指定可能。未設定時は空配列。 */
  const allowedHosts = env.VITE_ALLOWED_HOSTS
    ? env.VITE_ALLOWED_HOSTS.split(",").map((h) => h.trim())
    : [];
  return {
    // 本番ビルドは backend の /app/ 配下から配信されるため、資産の参照も /app/ 起点にする。
    // dev server は従来どおりルート配信（:3000/）なので base を変えない。
    base: command === "build" ? "/app/" : "/",
    plugins: [react()],
    server: {
      host: "0.0.0.0",
      port: 3000,
      allowedHosts,
      proxy: {
        "/api": "http://localhost:8000",
        "/v1": "http://localhost:8000",
        "/ui": "http://localhost:8000",
        // ファビコン等の共有静的資産は backend の /static にしか無い。
        // proxy に載せないと dev server の SPA フォールバックが index.html を返し、
        // ファビコンが「取得できたのに表示されない」状態になる。
        "/static": "http://localhost:8000",
      },
    },
  };
});
