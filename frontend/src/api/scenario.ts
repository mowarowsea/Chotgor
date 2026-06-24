/** バックエンドAPI呼び出し層 — シナリオチャット。 */

import { parseSSEStream } from "./sse";

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
  history_max_turns: number | null;
  history_max_chars: number | null;
  /** PC枠定義（engine_type="ensemble_pc" 用）。各枠は人物像・知っていること等を
   *  description に 1 テキストで持つ。セッション開始時に「誰が演じるか」を割り当てる。 */
  pc_slots?: PcSlot[];
  /** バナー画像の base64 data URI（オプション）。一覧・選択画面の見栄え用。 */
  banner_data?: string | null;
  created_at: string;
  updated_at: string;
}

/** シナリオに紐づく PC枠。AI キャラまたはユーザのどちらが演じるかは
 *  セッション側で割り当てる（`ScenarioSession.pc_assignments`）。 */
export interface PcSlot {
  slot_id: string;
  name: string;
  description: string;
  /** アバター画像の base64 data URI（オプション）。表示専用でプロンプトには載らない。 */
  image_data?: string | null;
}

/** セッション側の PC枠割当て。slot_id ごとに「ユーザ／AI キャラが演じる」を選ぶ。
 *
 *  - player_type="user": そのスロットをユーザ本人が担当する。
 *  - player_type="character": Chotgor の AI キャラが担当する。character_id 必須、
 *    preset_id 推奨（未指定ならキャラの enabled_providers 先頭がフォールバック）。
 */
export interface PcAssignment {
  slot_id: string;
  player_type: "user" | "character";
  character_id?: string;
  preset_id?: string;
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
  /** PC 配役一覧（engine_type="ensemble_pc" のみ。ensemble では空配列）。
   *  形式は新仕様: 各エントリはシナリオの pc_slots[slot_id] を「ユーザ／AI キャラ」に
   *  割り当てる。 */
  pc_assignments: PcAssignment[];
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
  /** debug_log_entries との紐付け。再生成ログをまとめるために使う。 */
  log_request_id?: string | null;
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

/**
 * あらすじ作成バー用の進捗情報。
 *
 * 前回あらすじ蒸留以降に積み上がった「ターン数（=話者ブロック数） / 文字数」とそれぞれの history 上限。
 * フロントは max(turns/max_turns, chars/max_chars) の比率でバーの表示可否（>50%）と
 * 色（>80% で赤）を決める。turn_complete 時とあらすじ作成（regenerate）時に更新される。
 */
export interface SynopsisProgress {
  turns: number;
  max_turns: number;
  chars: number;
  max_chars: number;
}

/** あらすじ作成（regenerate）APIのレスポンス。蒸留後の synopsis と最新の進捗。 */
export interface SynopsisRegenerateResult {
  synopsis: ScenarioSynopsis;
  progress: SynopsisProgress;
}

/** シナリオストリーミング SSE イベントの型定義。
 *
 * event 名は 1on1 の SSE（chunk/reasoning/error）と統一されている。
 * Scenario 固有の構造（複数話者ブロック・PC レスポンス連鎖・あらすじ進捗）は
 * turn_start/turn_end や user_saved / pc_done / turn_complete / synopsis_progress に集約。
 */
export type ScenarioStreamEvent =
  | { type: "user_saved"; turn: ScenarioTurn }
  // 新しい話者が発話を開始する通知。GM ターンは speaker_type/name/id/is_known、
  // PC ターン（ensemble_pc）は character/character_id を持つ。
  | {
      type: "turn_start";
      speaker_type: string;
      speaker_id: string | null;
      speaker_name: string;
      is_known: boolean;
    }
  | { type: "turn_start"; character: string; character_id: string }
  // 現在進行中の話者の本文チャンク。GM は text、PC は character + content。
  | { type: "chunk"; text: string }
  | { type: "chunk"; character: string; content: string }
  // 現在進行中の話者の reasoning（想起記憶・WM スレッド・思考ブロック）。1on1 と統一。
  | { type: "reasoning"; character: string; content: string }
  // 話者ブロック確定通知。turn は DB 保存済み ScenarioTurn。
  | { type: "turn_end"; turn: ScenarioTurn }
  // ── ensemble_pc 専用イベント ─────────────────────────────
  // PC レスポンス完了通知。full_text は最終応答、anticipation は ANTICIPATE_RESPONSE 抽出値。
  // log_message_id は 1on1 同様にバブルからログ画面へ飛ぶための 8 桁 hex（CHOTGOR_DEBUG=1 時のみ）。
  // この後 turn_end が来て turn が確定する。
  | {
      type: "pc_done";
      character: string;
      character_id: string;
      preset_name: string;
      full_text: string;
      anticipation: string | null;
      log_message_id?: string;
    }
  // アングル切替（switch_angle 経由）。GM/PC どちらでも発生しうるが現状は PC のみ。
  | {
      type: "angle_switched";
      character: string;
      character_id: string;
      model_id: string;
      preset_id: string;
      preset_name: string;
    }
  // ユーザターン完了（GM/PC のレスポンス連鎖が終わり、ユーザ入力待ちへ戻った）。
  // turn_ids は保存された話者ブロック ID、fired_responses は LLM 呼出回数（GM + PC）。
  | { type: "turn_complete"; turn_ids: string[]; fired_responses?: number }
  // ユーザターン完了直後のあらすじ進捗（ターン数=話者ブロック数・文字数と上限）。
  // フロントはこれでバーの表示/色と作成モーダルの自動表示を判定する。
  | {
      type: "synopsis_progress";
      turns: number;
      max_turns: number;
      chars: number;
      max_chars: number;
    }
  // エラー。GM 由来は message のみ、PC 由来は character/character_id も含む。
  | { type: "error"; message: string; character?: string; character_id?: string }
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
 *
 * `engineType` を "ensemble_pc" にした場合は `pcAssignments` を 1 件以上必須とする。
 * 既存の TRPG 進行に Chotgor キャラを PC として参加させる TRPG モード（GM + PCs）。
 * 省略時は "ensemble"（GM のみ・既存挙動）。
 */
export async function startScenarioSession(
  scenarioId: string,
  gmPresetId: string,
  synopsisPresetId: string,
  title?: string,
  engineType: "ensemble" | "ensemble_pc" = "ensemble",
  pcAssignments?: PcAssignment[],
): Promise<ScenarioSession> {
  const res = await fetch("/api/scenario_chat/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scenario_id: scenarioId,
      gm_preset_id: gmPresetId,
      synopsis_preset_id: synopsisPresetId,
      ...(title ? { title } : {}),
      engine_type: engineType,
      ...(engineType === "ensemble_pc" && pcAssignments
        ? { pc_assignments: pcAssignments }
        : {}),
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

/** あらすじを強制的に再蒸留する（ユーザ起動の「あらすじ作成」フロー）。
 *
 * 通常チャットでは閾値到達時に作成を促すだけで蒸留は走らせない。本関数が
 * その蒸留本体を起動する。既存 auto は**書き換えず**、新規分だけが末尾に追記される。
 *
 * `synopsisPresetId` を指定すると、その preset をセッションへ永続化（記憶）した上で
 * 蒸留に使う。あらすじ作成モーダルで選んだモデルが次回以降の既定にもなる。
 */
export async function regenerateScenarioSynopsis(
  sessionId: string,
  synopsisPresetId?: string,
): Promise<SynopsisRegenerateResult> {
  const res = await fetch(
    `/api/scenario_chat/sessions/${sessionId}/synopsis/regenerate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(
        synopsisPresetId ? { synopsis_preset_id: synopsisPresetId } : {},
      ),
    },
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? "あらすじの作成に失敗しました");
  }
  return res.json();
}

/**
 * ターンを SSE でストリーミング送信し、イベントを yield する。
 *
 * autoAdvance=true なら「ユーザは無言で続きを促す」モードで、
 * content は無視され、user turn も保存されない。
 * regenerateRequestId を指定すると、再生成ログを同一エントリにまとめる。
 * yieldTo は ensemble_pc の「ターンを譲る」UI 用（PC枠名 / "GM" / "ALL"）。
 * autoAdvance=true と組み合わせ、初動ルーティングをサーバ側で直接指定する。
 */
export async function* streamScenarioMessage(
  sessionId: string,
  content: string,
  autoAdvance: boolean = false,
  regenerateRequestId?: string,
  yieldTo?: string,
): AsyncGenerator<ScenarioStreamEvent> {
  const res = await fetch(
    `/api/scenario_chat/sessions/${sessionId}/stream`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content,
        auto_advance: autoAdvance,
        ...(regenerateRequestId
          ? { regenerate_request_id: regenerateRequestId }
          : {}),
        ...(yieldTo ? { yield_to: yieldTo } : {}),
      }),
    },
  );
  if (!res.ok) throw new Error("シナリオストリームの送信に失敗しました");
  yield* parseSSEStream<ScenarioStreamEvent>(res);
}

