/** バックエンドAPI呼び出し層 — デバッグログ閲覧（チャットバブルのログ折りたたみ用）。 */

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
  /** 実行結果（"ok" / "error"）。実行イベント由来のタグにのみ存在する
   *  （生ログ解析由来の過去ログタグでは実行成否が分からないため undefined）。 */
  status?: string;
  /** status="error" 時の失敗詳細。 */
  error_message?: string;
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

/** 1試行（メイン行 chat/scenario 等）の詳細データ。再生成すると同 request_id に複数 attempt が並ぶ。 */
export interface LogAttempt {
  index: number;
  preset: string;
  response: string;
  reasoning: string;
  dt_str: string;
  has_error: boolean;
  warn_reason: string;
  tool_calls: LogToolCall[];
  /** 実行イベント（tool_call_events）由来のツール使用（2026-06-11〜の新方式）。
   *  過去ログでは空で、代わりに tool_calls[].tags（生ログ解析）に値が入る。 */
  tags?: LogTag[];
  warnings: LogWarning[];
  files: string[];
  dir_id: string;
}

/** デバッグログの1リクエスト分のエントリ。
 *
 * `attempts` にメイン行（chat/scenario 等）の試行詳細が入る。
 * top-level の `tool_calls` / `warnings` は非メイン行（chronicle/forget 等、
 * メイン行が無いエントリ）の旧来互換フィールド。
 */
export interface LogEntry {
  /** リクエストID（8桁 hex）。シナリオの再生成では複数試行で共有される。 */
  message_id: string;
  /** 生ログフォルダ名（8桁 hex）。最新メイン行の raw_dir 由来で、再生成があると
   *  message_id（request_id）とは別の値になる。Raw ファイル取得にはこちらを使う。 */
  dir_id?: string;
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
  /** 実行イベント由来のツール使用（非メイン行エントリ用、LogAttempt.tags と同様）。 */
  tags?: LogTag[];
  warnings: LogWarning[];
  files: string[];
  has_error: boolean;
  /** メイン行の試行詳細リスト。再生成で複数試行ある場合は length > 1。 */
  attempts?: LogAttempt[];
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

