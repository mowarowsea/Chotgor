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
  /** デバッグログフォルダ名（8桁hex）。CHOTGOR_DEBUG=1 時のみ存在する。 */
  log_message_id?: string;
  created_at: string;
}

/** グループチャット設定オブジェクト（group_config のデシリアライズ後）。
 *
 * 司会モデルはセッション単位ではなくシステム設定で一括管理するため
 * group_config には含まれない（システム設定 ``group_director_preset_id`` で一元管理）。
 */
export interface GroupConfig {
  participants: Array<{ char_name: string; preset_name: string }>;
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
  /** 司会エラー（手動再試行・手動指名で復帰可能） */
  | { type: "director_error"; message: string }
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
  /** switch_angle 完了: 切り替え後の model_id（"{char_name}@{preset_name}" 形式）。
   *  次ターン以降のリクエストで使う selectedModel を更新するために使う。 */
  | { type: "angle_switched"; model_id: string }
  | { type: "done"; log_message_id?: string; user_message: ChatMessage; character_message: ChatMessage }
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

/** グループチャットセッションを作成する。
 *
 * 司会モデルはシステム設定（Settings画面）で一括管理するため引数に含まない。
 */
export async function createGroupSession(
  participants: string[],
  maxAutoTurns: number,
  turnTimeoutSec: number,
): Promise<Session & { warning?: string }> {
  const res = await fetch("/api/group/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      participants: participants.map((id) => ({ model_id: id })),
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
 * @param targetCharacter - 指定した場合、司会を介さずそのキャラクターを手動指名して発言させる。
 */
export async function* streamGroupMessage(
  sessionId: string,
  content: string,
  imageIds?: string[],
  skip?: boolean,
  targetCharacter?: string | null,
): AsyncGenerator<GroupStreamEvent> {
  const res = await fetch(`/api/group/sessions/${sessionId}/messages/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      content,
      ...(imageIds && imageIds.length > 0 ? { image_ids: imageIds } : {}),
      ...(skip ? { skip: true } : {}),
      ...(targetCharacter ? { target_character: targetCharacter } : {}),
    }),
  });
  if (!res.ok) throw new Error("グループメッセージの送信に失敗しました");
  yield* parseSSEStream<GroupStreamEvent>(res);
}

// ---------------------------------------------------------------------------
// シナリオチャット
// ---------------------------------------------------------------------------

/** シナリオテンプレート — 何度でも遊べる設定の塊。backend UI で登録・編集する。
 *
 * 場所・空気感・語り口・テンポなどの細分情報はすべて `scenario` テキストにまとめる
 * （以前は別フィールドだったが、線引きが曖昧だったので統合）。
 *
 * GM の LLM プリセットはテンプレートではなくセッション単位（`ScenarioSession.gm_preset_id`）
 * で保持する。同一シナリオから複数セッションを起動した際にそれぞれ別の GM モデルで
 * 遊べるようにするため。
 */
export interface ScenarioTemplate {
  id: string;
  title: string;
  scenario: string | null;
  intro: string | null;
  user_alias: string;
  history_max_turns: number | null;
  history_max_chars: number | null;
  created_at: string;
  updated_at: string;
}

/** シナリオテンプレートに紐づく NPC。
 *
 * description には人物像・口調・話し方を自由テキストで全部詰め込む。
 * image_data はアバター画像の base64 data URI（オプション）。
 */
export interface ScenarioNpc {
  id: string;
  scenario_id: string;
  name: string;
  description: string | null;
  image_data: string | null;
  promoted_character_id: string | null;
  created_at: string;
}

/** シナリオから起動されたプレイインスタンス。フロントの「セッション一覧」に並ぶ。
 *
 * session_type は判別用にフロント側で付与する。
 */
export interface ScenarioSession {
  id: string;
  scenario_id: string;
  title: string;
  engine_type: string;
  status: string;
  /** GM が使う LLM プリセット ID。セッション開始時に必須・チャット中も変更可。 */
  gm_preset_id: string;
  /** あらすじ蒸留専用の LLM プリセット ID。GM とは別モデルを指定可能（レートリミット節約）。
   *  セッション開始時に必須・チャット中も同モーダルから変更可。
   */
  synopsis_preset_id: string;
  created_at: string;
  updated_at: string;
  /** フロント側で判別用に追加（バックエンドからは返らない）。 */
  session_type: "scenario";
}

/** シナリオセッションの発話ターン。 */
export interface ScenarioTurn {
  id: string;
  session_id: string;
  turn_index: number;
  /** "user" | "narrator" | "npc" | "character" */
  speaker_type: string;
  /** known NPC のみ NPC.id が入る。Narrator・未知 NPC・user は null。 */
  speaker_id: string | null;
  /** 表示用スナップショット名。 */
  speaker_name: string;
  content: string;
  raw_response: string | null;
  created_at: string;
}

/** プレイセッション詳細（元シナリオ + NPC を含む）。 */
export interface ScenarioSessionDetail extends ScenarioSession {
  scenario: ScenarioTemplate | null;
  npcs: ScenarioNpc[];
}

/** セッション単位のあらすじ（記憶捏造対策）。
 *
 * auto: LLM 自動生成（追記専用）。GM プロンプトのメインのあらすじ。
 * manual: プレイヤー手書きの補足メモ。自動更新では一切触らない。
 * last_turn_index: synopsis_auto に「どこまで要約済みか」を記録する境界。
 */
export interface ScenarioSynopsis {
  auto: string;
  manual: string;
  last_turn_index: number;
}

/** シナリオストリーミング SSE イベントの型定義。 */
export type ScenarioStreamEvent =
  | { type: "user_saved"; turn: ScenarioTurn }
  | {
      type: "speaker_start";
      speaker_type: string;
      speaker_id: string | null;
      speaker_name: string;
      is_known: boolean;
    }
  | { type: "content_delta"; text: string }
  | { type: "speaker_end"; turn: ScenarioTurn }
  | { type: "turn_complete"; turn_ids: string[] }
  | { type: "synopsis_updated"; synopsis: ScenarioSynopsis }
  | { type: "error"; message: string }
  | { type: "done" };

/** タグ session_type を付与するヘルパ。 */
function tagScenarioSession(
  s: Omit<ScenarioSession, "session_type">,
): ScenarioSession {
  return { ...s, session_type: "scenario" as const };
}

// ─── シナリオテンプレート（一覧取得のみ。CRUD は backend UI で行う） ──────

/** シナリオテンプレート一覧を取得する。フロントでは「開始するシナリオの選択」のみ使う。 */
export async function fetchScenarioTemplates(): Promise<ScenarioTemplate[]> {
  const res = await fetch("/api/scenario_chat/scenarios");
  if (!res.ok) throw new Error("シナリオテンプレート一覧の取得に失敗しました");
  return res.json();
}

/** GM プリセット情報。シナリオの `gm_preset_id` を表示名に解決するために使う。 */
export interface ScenarioPreset {
  id: string;
  name: string;
  provider: string;
  model_id: string;
  thinking_level: string;
}

/** GM プリセット一覧を取得する。シナリオヘッダーの `gm_preset_id` → 表示名解決に使う。 */
export async function fetchScenarioPresets(): Promise<ScenarioPreset[]> {
  const res = await fetch("/api/scenario_chat/presets");
  if (!res.ok) throw new Error("GM プリセット一覧の取得に失敗しました");
  return res.json();
}

// ─── プレイセッション ──────────────────────────────────────────────────────

/** プレイセッション一覧を取得する。 */
export async function fetchScenarioSessions(): Promise<ScenarioSession[]> {
  const res = await fetch("/api/scenario_chat/sessions");
  if (!res.ok) throw new Error("シナリオセッション一覧の取得に失敗しました");
  const data = (await res.json()) as Array<Omit<ScenarioSession, "session_type">>;
  return data.map(tagScenarioSession);
}

/** プレイセッション詳細を取得する（元シナリオ・NPC を含む）。 */
export async function fetchScenarioSession(
  sessionId: string,
): Promise<ScenarioSessionDetail> {
  const res = await fetch(`/api/scenario_chat/sessions/${sessionId}`);
  if (!res.ok) throw new Error("シナリオセッションの取得に失敗しました");
  const data = (await res.json()) as Omit<ScenarioSession, "session_type"> & {
    scenario: ScenarioTemplate | null;
    npcs: ScenarioNpc[];
  };
  return {
    ...tagScenarioSession(data),
    scenario: data.scenario,
    npcs: data.npcs,
  };
}

/** プレイセッションのターン一覧を取得する。 */
export async function fetchScenarioTurns(sessionId: string): Promise<ScenarioTurn[]> {
  const res = await fetch(`/api/scenario_chat/sessions/${sessionId}/turns`);
  if (!res.ok) throw new Error("シナリオターンの取得に失敗しました");
  return res.json();
}

/** シナリオから新しいプレイセッションを起動する。
 *
 * `gmPresetId` はこのセッションで GM を演じる LLM プリセット ID。
 * `synopsisPresetId` はあらすじ蒸留専用の LLM プリセット ID（同じプリセットでもよい）。
 * セッション開始後も左上ヘッダーのモーダルから両方変更可能。
 */
export async function startScenarioSession(
  scenarioId: string,
  gmPresetId: string,
  synopsisPresetId: string,
  title?: string,
): Promise<ScenarioSession> {
  const res = await fetch("/api/scenario_chat/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scenario_id: scenarioId,
      gm_preset_id: gmPresetId,
      synopsis_preset_id: synopsisPresetId,
      ...(title ? { title } : {}),
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? "シナリオセッションの起動に失敗しました");
  }
  const data = (await res.json()) as Omit<ScenarioSession, "session_type">;
  return tagScenarioSession(data);
}

/** プレイセッションを部分更新する（タイトル / status / GM モデル / あらすじモデル）。 */
export async function updateScenarioSession(
  sessionId: string,
  patch: {
    title?: string;
    status?: string;
    gm_preset_id?: string;
    synopsis_preset_id?: string;
  },
): Promise<ScenarioSession> {
  const res = await fetch(`/api/scenario_chat/sessions/${sessionId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? "シナリオセッションの更新に失敗しました");
  }
  const data = (await res.json()) as Omit<ScenarioSession, "session_type">;
  return tagScenarioSession(data);
}

/** プレイセッションを削除する。テンプレには影響しない。 */
export async function deleteScenarioSession(sessionId: string): Promise<void> {
  const res = await fetch(`/api/scenario_chat/sessions/${sessionId}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error("シナリオセッションの削除に失敗しました");
}

/** プレイセッションを終了する（status=ended）。 */
export async function endScenarioSession(
  sessionId: string,
): Promise<ScenarioSession> {
  const res = await fetch(`/api/scenario_chat/sessions/${sessionId}/end`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("シナリオセッションの終了に失敗しました");
  const data = (await res.json()) as Omit<ScenarioSession, "session_type">;
  return tagScenarioSession(data);
}

/** 指定ターン以降（自身を含む）をすべて削除する。編集・再生成の前処理。 */
export async function deleteScenarioTurnsFrom(
  sessionId: string,
  turnId: string,
): Promise<void> {
  const res = await fetch(
    `/api/scenario_chat/sessions/${sessionId}/turns/from/${turnId}`,
    { method: "DELETE" },
  );
  if (!res.ok) throw new Error("ターンの削除に失敗しました");
}

// ─── あらすじ（記憶捏造対策） ──────────────────────────────────────────────

/** セッションのあらすじ（auto / manual / last_turn_index）を取得する。 */
export async function fetchScenarioSynopsis(
  sessionId: string,
): Promise<ScenarioSynopsis> {
  const res = await fetch(
    `/api/scenario_chat/sessions/${sessionId}/synopsis`,
  );
  if (!res.ok) throw new Error("あらすじの取得に失敗しました");
  return res.json();
}

/** セッションのあらすじを部分更新する。
 *
 * `auto` と `manual` はそれぞれ独立に更新可能（undefined のフィールドは触らない）。
 * ユーザが捏造記述を発見した場合、UI 上で `auto` を直接編集して送信できる。
 */
export async function patchScenarioSynopsis(
  sessionId: string,
  patch: { auto?: string; manual?: string },
): Promise<ScenarioSynopsis> {
  const res = await fetch(
    `/api/scenario_chat/sessions/${sessionId}/synopsis`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    },
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? "あらすじの更新に失敗しました");
  }
  return res.json();
}

/** `synopsis_auto` への自動追記フローを手動で起動する。
 *
 * 通常チャットでは閾値未満は自動更新を抑制しているため、ユーザが任意の
 * タイミングで強制追記したい場合に使う。既存 auto は**書き換えず**、
 * 新規分だけが末尾に追記される。
 */
export async function regenerateScenarioSynopsis(
  sessionId: string,
): Promise<ScenarioSynopsis> {
  const res = await fetch(
    `/api/scenario_chat/sessions/${sessionId}/synopsis/regenerate`,
    { method: "POST" },
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? "あらすじの再生成に失敗しました");
  }
  return res.json();
}

/**
 * ターンを SSE でストリーミング送信し、イベントを yield する。
 *
 * autoAdvance=true なら「ユーザは無言で続きを促す」モードで、
 * content は無視され、user turn も保存されない。
 */
export async function* streamScenarioMessage(
  sessionId: string,
  content: string,
  autoAdvance: boolean = false,
): AsyncGenerator<ScenarioStreamEvent> {
  const res = await fetch(
    `/api/scenario_chat/sessions/${sessionId}/stream`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content, auto_advance: autoAdvance }),
    },
  );
  if (!res.ok) throw new Error("シナリオストリームの送信に失敗しました");
  yield* parseSSEStream<ScenarioStreamEvent>(res);
}

// ---------------------------------------------------------------------------
// ログ閲覧（チャットバブル用）
// ---------------------------------------------------------------------------

/** タグのバッジ表示メタ情報。 */
export interface LogTagMeta {
  label: string;
  cls: string;
}

/** デバッグログから抽出されたタグ1件。 */
export interface LogTag {
  tag_name: string;
  meta: LogTagMeta;
  fields: Record<string, string>;
  preview: string;
}

/** ツール呼び出し（LLMプロバイダーへの1回のRequest/Responseペア）。 */
export interface LogToolCall {
  feature: string;
  preset: string;
  request_file: string | null;
  response_file: string | null;
  tags: LogTag[];
}

/** ログの警告エントリ。 */
export interface LogWarning {
  tag: string;
  message: string;
  file: string;
}

/** デバッグログの1リクエスト分のエントリ。 */
export interface LogEntry {
  /** debug フォルダ名（8桁 hex）。 */
  message_id: string;
  dt_str: string;
  character: string;
  preset: string;
  model_id: string;
  source: string;
  user_message: string;
  character_response: string;
  /** 思考ブロック・想起記憶テキスト。CHOTGOR_DEBUG=1 かつ Thinking 有効時のみ存在する。 */
  reasoning_text?: string;
  tool_calls: LogToolCall[];
  warnings: LogWarning[];
  files: string[];
  has_error: boolean;
}

/**
 * 指定した log_message_id のデバッグログエントリを取得する。
 * CHOTGOR_DEBUG=1 が設定されていない場合は 404 になる。
 */
export async function fetchLogEntry(logMessageId: string): Promise<LogEntry> {
  const res = await fetch(`/api/logs/entry/${encodeURIComponent(logMessageId)}`);
  if (!res.ok) throw new Error("ログエントリの取得に失敗しました");
  const data = await res.json();
  return data.entry as LogEntry;
}

/** 指定リクエストの生ログファイル内容を取得する。 */
export async function fetchRawLog(messageId: string, filename: string): Promise<string> {
  const res = await fetch(`/ui/logs/${encodeURIComponent(messageId)}/raw/${encodeURIComponent(filename)}`);
  if (!res.ok) throw new Error("ログファイルの取得に失敗しました");
  return res.text();
}

// ---------------------------------------------------------------------------
// 翻訳
// ---------------------------------------------------------------------------

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
