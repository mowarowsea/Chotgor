/**
 * 会話エクスポート用フォーマット管理フック。
 * テンプレート文字列を localStorage に永続化し、Markdown 出力を生成する。
 */
import { useState, useCallback } from "react";
import type { ChatMessage } from "../api";

// ---------------------------------------------------------------------------
// デフォルトテンプレート
// ---------------------------------------------------------------------------

export const DEFAULT_CHARACTER_TEMPLATE =
  "**{character_name}**:\n\n> {reasoning}\n\n{content}";

export const DEFAULT_USER_TEMPLATE = "**{user_name}**:\n\n{content}";

const STORAGE_KEY = "chotgor_export_format";

// ---------------------------------------------------------------------------
// 型定義
// ---------------------------------------------------------------------------

export interface ExportFormat {
  characterTemplate: string;
  userTemplate: string;
}

// ---------------------------------------------------------------------------
// ユーティリティ
// ---------------------------------------------------------------------------

function loadFormat(): ExportFormat {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw) as ExportFormat;
  } catch {
    // 読み込み失敗時はデフォルトを返す
  }
  return {
    characterTemplate: DEFAULT_CHARACTER_TEMPLATE,
    userTemplate: DEFAULT_USER_TEMPLATE,
  };
}

/**
 * テンプレート文字列に変数を埋め込んでレンダリングする。
 * `includeReasoning` が false または reasoning が空のとき、
 * `{reasoning}` を含む行ごと削除して連続する空行を圧縮する。
 */
export function renderMessage(
  template: string,
  vars: {
    character_name?: string;
    user_name?: string;
    content: string;
    reasoning?: string;
    timestamp?: string;
  },
  includeReasoning: boolean,
): string {
  let result = template;
  result = result.replace(/\{character_name\}/g, vars.character_name ?? "");
  result = result.replace(/\{user_name\}/g, vars.user_name ?? "");
  result = result.replace(/\{content\}/g, vars.content);
  result = result.replace(/\{timestamp\}/g, vars.timestamp ?? "");

  if (includeReasoning && vars.reasoning) {
    result = result.replace(/\{reasoning\}/g, vars.reasoning);
  } else {
    // {reasoning} を含む行を除去し、3行以上の連続空行を2行に圧縮
    result = result
      .split("\n")
      .filter((line) => !line.includes("{reasoning}"))
      .join("\n")
      .replace(/\n{3,}/g, "\n\n");
  }

  return result.trim();
}

/**
 * メッセージ一覧を Markdown テキストにエクスポートする。
 * セッションタイトルが指定されていれば先頭に `# タイトル` を付与する。
 */
export function buildExportText(
  messages: ChatMessage[],
  reasoningMap: Record<string, string>,
  userName: string,
  format: ExportFormat,
  includeReasoning: boolean,
  sessionTitle?: string,
): string {
  const parts: string[] = [];

  if (sessionTitle) {
    parts.push(`# ${sessionTitle}`);
  }

  for (const msg of messages) {
    if (msg.role === "user") {
      parts.push(
        renderMessage(
          format.userTemplate,
          {
            user_name: userName,
            content: msg.content,
            timestamp: msg.created_at,
          },
          includeReasoning,
        ),
      );
    } else {
      const reasoning = reasoningMap[msg.id] ?? msg.reasoning;
      parts.push(
        renderMessage(
          format.characterTemplate,
          {
            character_name: msg.character_name ?? "キャラクター",
            content: msg.content,
            reasoning,
            timestamp: msg.created_at,
          },
          includeReasoning,
        ),
      );
    }
  }

  return parts.join("\n\n---\n\n");
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/** エクスポートフォーマットを localStorage で管理するフック。 */
export function useExportFormat() {
  const [format, setFormat] = useState<ExportFormat>(loadFormat);

  const updateFormat = useCallback((next: Partial<ExportFormat>) => {
    setFormat((prev) => {
      const updated = { ...prev, ...next };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      return updated;
    });
  }, []);

  const resetFormat = useCallback(() => {
    const defaults: ExportFormat = {
      characterTemplate: DEFAULT_CHARACTER_TEMPLATE,
      userTemplate: DEFAULT_USER_TEMPLATE,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(defaults));
    setFormat(defaults);
  }, []);

  return { format, updateFormat, resetFormat };
}
