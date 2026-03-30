/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
      },
      colors: {
        /* Chotgor デザインシステム：モノクローム + ジェードアクセント */
        ch: {
          bg:    "#090909",       // ほぼ純黒
          s1:    "#202020",       // 第1サーフェス
          s2:    "#292929",       // 第2サーフェス（ホバー等）
          s3:    "#333333",       // 第3サーフェス（インプット内等）
          accent: "#4d8c67",      // ジェードグリーン — アクセントのみ
          "accent-t": "#6aaa84",  // アクセント文字色（やや明るめ）
          "accent-dim": "#162519",// 暗いアクセント背景（ボタン用）
          t1:    "#e4e4e4",       // 最強テキスト
          t2:    "#8c8c8c",       // 中間テキスト
          t3:    "#505050",       // 弱テキスト
          t4:    "#2e2e2e",       // 最弱テキスト（区切り・ラベル等）
        },
      },
    },
  },
  plugins: [],
};
