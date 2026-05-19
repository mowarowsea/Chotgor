/**
 * ライト/ダークテーマの状態管理フック。
 *
 * テーマは <html> 要素の data-dark 属性で表現し、index.css の CSS 変数を切り替える。
 * 選択は localStorage に永続化し、未設定時は OS の配色設定（prefers-color-scheme）に従う。
 */
import { useCallback, useEffect, useState } from "react";

/** localStorage に保存する際のキー。 */
const STORAGE_KEY = "chotgor-theme";

/** 保存値または OS 設定から初期テーマ（ダークかどうか）を解決する。 */
function resolveInitialDark(): boolean {
  if (typeof window === "undefined") return false;
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved === "dark") return true;
  if (saved === "light") return false;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
}

/**
 * テーマ管理フック。
 * @returns dark - 現在ダークテーマかどうか / toggle - テーマを反転する関数。
 */
export function useTheme(): { dark: boolean; toggle: () => void } {
  const [dark, setDark] = useState<boolean>(resolveInitialDark);

  // dark の変化を <html data-dark> 属性と localStorage に反映する。
  useEffect(() => {
    document.documentElement.toggleAttribute("data-dark", dark);
    localStorage.setItem(STORAGE_KEY, dark ? "dark" : "light");
  }, [dark]);

  /** テーマを反転する。 */
  const toggle = useCallback(() => setDark((d) => !d), []);

  return { dark, toggle };
}
