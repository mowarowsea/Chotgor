/** バックエンドAPI呼び出し層。 */

// ---------------------------------------------------------------------------
// モデルID ユーティリティ
// ---------------------------------------------------------------------------

/**
 * モデルID（"{char_name}@{preset_name}" 形式）からキャラクター名を抽出する。
 *
 * Sidebar.tsx・App.tsx など複数箇所で使われるため api.ts に一元化する。
 *
 * @example charNameOf("Alice@thinking")  // => "Alice"
 */
export function charNameOf(modelId: string): string {
  return modelId.split("@")[0];
}

/**
 * モデルID（"{char_name}@{preset_name}" 形式）からプリセット名を抽出する。
 *
 * @example presetNameOf("Alice@thinking")  // => "thinking"
 */
export function presetNameOf(modelId: string): string {
  return modelId.split("@")[1] ?? "";
}

export interface Model {
  id: string;   // "{char_name}@{preset_name}"
  object: string;
}

export interface Session {
  id: string;
  model_id: string;
  title: string;
  /** セッション種別。"1on1" または "group"。 */
  session_type: "1on1" | "group";
  /** グループチャット設定JSON文字列。session_type="group" のみ存在する。 */
  group_config?: string;
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
  /** 添付画像IDのリスト。ユーザメッセージのみ存在する場合がある。 */
  images?: string[];
  /** グループチャット時の発言キャラクター名。 */
  character_name?: string;
  /** メッセージ送信時に使用したプリセット名（バブル表示用）。 */
  preset_name?: string;
  created_at: string;
}

/** グループチャット設定オブジェクト（group_config のデシリアライズ後）。 */
export interface GroupConfig {
  participants: Array<{ char_name: string; preset_name: string }>;
  director_model_id: string;  // "{char_name}@{preset_name}" 形式
  max_auto_turns: number;
  turn_timeout_sec: number;
}

/** グループチャットSSEイベントの型定義。 */
export type GroupStreamEvent =
  | { type: "user_saved"; message: ChatMessage }
  | { type: "speaker_decided"; speakers: string[] }
  /** キャラクター応答開始（スピナー表示用） */
  | { type: "character_start"; character: string }
  /** 思考ブロック・想起記憶（リアルタイムストリーミング） */
  | { type: "character_reasoning"; character: string; content: string }
  /** 応答テキスト（1チャンク） */
  | { type: "character_chunk"; character: string; content: string }
  /** DB保存完了（確定済みメッセージ） */
  | { type: "character_done"; character: string; message: ChatMessage }
  | { type: "user_turn"; auto_turns_used: number }
  | { type: "error"; message: string; character?: string }
  | { type: "done" };

export interface SessionDetail extends Session {
  messages: ChatMessage[];
}

/** キャラクターの型定義。 */
export interface Character {
  id: string;
  name: string;
  /** Afterglow（感情継続機構）の新規チャット作成時デフォルト値。 */
  afterglow_default: boolean;
}

/** キャラクター一覧を取得する。 */
export async function fetchCharacters(): Promise<Character[]> {
  const res = await fetch("/api/characters/");
  if (!res.ok) throw new Error("キャラクター一覧の取得に失敗しました");
  return res.json();
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

/**
 * 新しいセッションを作成する。
 *
 * @param modelId - "{char_name}@{preset_name}" 形式のモデルID。
 * @param afterglow - Afterglow（感情継続機構）を有効にする場合は true。
 *                   true の場合、同キャラクターの直近5ターンを引き継いでLLMに渡す。
 */
export async function createSession(modelId: string, afterglow = false): Promise<Session> {
  const res = await fetch("/api/chat/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, afterglow }),
  });
  if (!res.ok) throw new Error("セッションの作成に失敗しました");
  return res.json();
}

/** セッションを削除する。 */
export async function deleteSession(sessionId: string): Promise<void> {
  const res = await fetch(`/api/chat/sessions/${sessionId}`, { method: "DELETE" });
  if (!res.ok) throw new Error("セッションの削除に失敗しました");
}

/** セッションのタイトルを更新する。 */
export async function updateSessionTitle(sessionId: string, title: string): Promise<Session> {
  const res = await fetch(`/api/chat/sessions/${sessionId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error("タイトルの更新に失敗しました");
  return res.json();
}

/** SSEストリームイベントの型定義。 */
export type StreamEvent =
  | { type: "chunk"; content: string }
  /** 思考ブロック・想起した記憶（フロントで折りたたみ表示する） */
  | { type: "reasoning"; content: string }
  /** switch_angle 発動: 表示をクリアして第2プロバイダーのストリームを開始する */
  | { type: "clear" }
  | { type: "done"; user_message: ChatMessage; character_message: ChatMessage }
  | { type: "error"; message: string };

/** SSEレスポンスボディを解析してイベントオブジェクトをyieldする共通ジェネレーター。 */
async function* parseSSEStream<T>(res: Response): AsyncGenerator<T> {
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
          yield JSON.parse(line.slice(6)) as T;
        } catch {
          // 不正なJSONはスキップ
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/** メッセージをSSEでストリーミング送信し、イベントをyieldする。 */
export async function* streamMessage(
  sessionId: string,
  content: string,
  imageIds?: string[],
  modelId?: string
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`/api/chat/sessions/${sessionId}/messages/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      content,
      ...(imageIds && imageIds.length > 0 ? { image_ids: imageIds } : {}),
      ...(modelId ? { model_id: modelId } : {}),
    }),
  });

  if (!res.ok) throw new Error("ストリーミング送信に失敗しました");
  yield* parseSSEStream<StreamEvent>(res);
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

/** 複数の画像ファイルをアップロードしてセッションに紐づける。 */
export async function uploadImages(
  sessionId: string,
  files: File[]
): Promise<{ id: string; url: string }[]> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  const res = await fetch(`/api/chat/sessions/${sessionId}/images`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error("画像のアップロードに失敗しました");
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

/** グループチャットセッションを作成する。 */
export async function createGroupSession(
  participants: string[],
  directorModelId: string,
  maxAutoTurns: number,
  turnTimeoutSec: number,
): Promise<Session & { warning?: string }> {
  const res = await fetch("/api/group/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      participants: participants.map((id) => ({ model_id: id })),
      director_model_id: directorModelId,
      max_auto_turns: maxAutoTurns,
      turn_timeout_sec: turnTimeoutSec,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? "グループセッションの作成に失敗しました");
  }
  return res.json();
}

/** SELF_DRIFT指針の型定義。 */
export interface Drift {
  id: string;
  session_id: string;
  character_id: string;
  content: string;
  enabled: boolean;
  created_at: string;
}

/** セッションの全SELF_DRIFT一覧を取得する。 */
export async function fetchDrifts(sessionId: string): Promise<Drift[]> {
  const res = await fetch(`/api/chat/sessions/${sessionId}/drifts`);
  if (!res.ok) throw new Error("SELF_DRIFT一覧の取得に失敗しました");
  return res.json();
}

/** SELF_DRIFT の enabled フラグを反転する。 */
export async function toggleDrift(sessionId: string, driftId: string): Promise<Drift> {
  const res = await fetch(`/api/chat/sessions/${sessionId}/drifts/${driftId}/toggle`, {
    method: "PATCH",
  });
  if (!res.ok) throw new Error("SELF_DRIFT のトグルに失敗しました");
  return res.json();
}

/** 指定キャラの全SELF_DRIFTを削除する。 */
export async function resetDrifts(sessionId: string, characterId: string): Promise<void> {
  const res = await fetch(
    `/api/chat/sessions/${sessionId}/drifts?character_id=${encodeURIComponent(characterId)}`,
    { method: "DELETE" }
  );
  if (!res.ok) throw new Error("SELF_DRIFT のリセットに失敗しました");
}

/** グループチャットメッセージをSSEでストリーミング送信し、イベントをyieldする。
 *
 * @param skip - true の場合、ユーザメッセージを保存せず司会へ直接ターンを委譲する（ユーザターンスキップ）。
 */
export async function* streamGroupMessage(
  sessionId: string,
  content: string,
  imageIds?: string[],
  skip?: boolean,
): AsyncGenerator<GroupStreamEvent> {
  const res = await fetch(`/api/group/sessions/${sessionId}/messages/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      content,
      ...(imageIds && imageIds.length > 0 ? { image_ids: imageIds } : {}),
      ...(skip ? { skip: true } : {}),
    }),
  });
  if (!res.ok) throw new Error("グループメッセージの送信に失敗しました");
  yield* parseSSEStream<GroupStreamEvent>(res);
}
