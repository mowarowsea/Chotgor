/** バックエンドAPI呼び出し層 — テキスト翻訳。 */

/**
 * テキストを日本語に翻訳する。
 *
 * Settings 画面で設定した翻訳モデルを使ってサーバ側で翻訳を実行する。
 * 翻訳結果はDBに保存されず、その場限りで表示されるのみ。
 *
 * @param text - 翻訳するテキスト。
 * @param presetId - 使用するモデルプリセットID（省略時はサーバ設定を使用）。
 * @returns 翻訳されたテキスト。
 * @throws APIサービスエラー(ステータスコード) またはその他のエラー。
 */
export async function translateText(text: string, presetId?: string): Promise<string> {
  const res = await fetch("/api/translate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, ...(presetId ? { preset_id: presetId } : {}) }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `APIサービスエラー(${res.status})`);
  }
  const data = await res.json();
  return data.translation as string;
}
