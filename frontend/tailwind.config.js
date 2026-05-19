/** @type {import('tailwindcss').Config} */

/**
 * CSS 変数（空白区切り RGB チャンネル）を Tailwind のカラートークンへ変換するヘルパー。
 * `<alpha-value>` プレースホルダにより `bg-ch-bg/95` のような透明度修飾子も機能する。
 */
function ch(name) {
  return `rgb(var(--ch-${name}) / <alpha-value>)`;
}

module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  /* ダークテーマは <html data-dark> 属性で切り替えるため selector 戦略を使う。 */
  darkMode: ["selector", "[data-dark]"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["DM Sans", "system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        mono: ["DM Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      colors: {
        /* Chotgor デザインシステム：ブルーグレー基調 + ブルーアクセント。
           実体は index.css の CSS 変数で、ライト/ダークは <html data-dark> で切り替わる。 */
        ch: {
          bg:    ch("bg"),        // 背景
          s1:    ch("s1"),        // 第1サーフェス（サイドバー等）
          s2:    ch("s2"),        // 第2サーフェス（ホバー等）
          s3:    ch("s3"),        // 第3サーフェス（インプット内・パネル）
          /* アクセントはライト/ダーク共通のブルー。oklch でも <alpha-value> が機能する。 */
          accent:       "oklch(50% 0.13 226 / <alpha-value>)",
          "accent-t":   "oklch(50% 0.13 226 / <alpha-value>)",
          "accent-dim": ch("accent-dim"),  // 淡いアクセント背景（ボタン用）
          t1:    ch("t1"),        // 最強テキスト
          t2:    ch("t2"),        // 中間テキスト
          t3:    ch("t3"),        // 弱テキスト
          t4:    ch("t4"),        // 最弱テキスト（区切り・ラベル等）
          ub:    ch("ub"),        // ユーザバブル背景
          ut:    ch("ut"),        // ユーザバブル文字色
          "t-emphasis":        ch("emphasis"),         // 強調（斜体）テキスト
          "t-emphasis-mobile": ch("emphasis-mobile"),  // 強調（斜体）テキスト・モバイル用
        },
      },
    },
  },
  plugins: [],
};
