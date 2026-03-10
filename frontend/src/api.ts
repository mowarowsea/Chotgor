/** バックエンドAPI呼び出し層。 */

export interface Model {
  id: string;   // "{char_name}@{preset_name}"
  object: string;
}

export interface Session {
  id: string;
  model_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ChatMessage {
  id: string;
  session_id: string;
  role: "user" | "character";
  content: string;
  /** 思考ブロック・想起記憶テキスト。キャラクターメッセージのみ存在する場合がある。 */
  reasoning?: string;
  created_at: string;
}

export interface SessionDetail extends Session {
  messages: ChatMessage[];
}

/** 利用可能なモデル（character@preset）一覧を取得する。 */
export async function fetchModels(): Promise<Model[]> {
  const res = await fetch("/v1/models");
  if (!res.ok) throw new Error("モデル一覧の取得に失敗しました");
  const data = await res.json();
  return data.data as Model[];
}

/** チャットセッション一覧を取得する。 */
export async function fetchSessions(): Promise<Session[]> {
  const res = await fetch("/api/chat/sessions");
  if (!res.ok) throw new Error("セッション一覧の取得に失敗しました");
  return res.json();
}

/** セッションとメッセージ一覧を取得する。 */
export async function fetchSession(sessionId: string): Promise<SessionDetail> {
  const res = await fetch(`/api/chat/sessions/${sessionId}`);
  if (!res.ok) throw new Error("セッションの取得に失敗しました");
  return res.json();
}

/** 新しいセッションを作成する。 */
export async function createSession(modelId: string): Promise<Session> {
  const res = await fetch("/api/chat/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId }),
  });
  if (!res.ok) throw new Error("セッションの作成に失敗しました");
  return res.json();
}

/** セッションを削除する。 */
export async function deleteSession(sessionId: string): Promise<void> {
  const res = await fetch(`/api/chat/sessions/${sessionId}`, { method: "DELETE" });
  if (!res.ok) throw new Error("セッションの削除に失敗しました");
}

/** SSEストリームイベントの型定義。 */
export type StreamEvent =
  | { type: "chunk"; content: string }
  /** 思考ブロック・想起した記憶（フロントで折りたたみ表示する） */
  | { type: "reasoning"; content: string }
  | { type: "done"; user_message: ChatMessage; character_message: ChatMessage }
  | { type: "error"; message: string };

/** メッセージをSSEでストリーミング送信し、イベントをyieldする。 */
export async function* streamMessage(
  sessionId: string,
  content: string
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`/api/chat/sessions/${sessionId}/messages/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });

  if (!res.ok) throw new Error("ストリーミング送信に失敗しました");

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          yield JSON.parse(line.slice(6)) as StreamEvent;
        } catch {
          // 不正なJSONはスキップ
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/** メッセージを送信してキャラクターの応答を受け取る。 */
export async function sendMessage(
  sessionId: string,
  content: string
): Promise<{ user_message: ChatMessage; character_message: ChatMessage }> {
  const res = await fetch(`/api/chat/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
  if (!res.ok) throw new Error("メッセージの送信に失敗しました");
  return res.json();
}

/** 指定メッセージ以降（自身を含む）をすべて削除する。編集・再生成の前処理に使う。 */
export async function deleteMessagesFrom(sessionId: string, messageId: string): Promise<void> {
  const res = await fetch(`/api/chat/sessions/${sessionId}/messages/from/${messageId}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error("メッセージの削除に失敗しました");
}

/** ユーザ名を取得する。 */
export async function fetchUserName(): Promise<string> {
  const res = await fetch("/api/chat/settings/user-name");
  if (!res.ok) return "ユーザ";
  const data = await res.json();
  return data.user_name ?? "ユーザ";
}
