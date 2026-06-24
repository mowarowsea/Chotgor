/**
 * 1on1チャットビューコンポーネント。
 * MessageList・MessageInput を組み合わせてレイアウトするだけのシェルコンポーネント。
 * メッセージ描画・入力フォームのロジックはそれぞれの子コンポーネントに委譲する。
 *
 * 対面モード時:
 *   - 外側ラッパに background-image を当てて「対面の場」を可視化する。
 *   - 背景画像は contain で全画面に縮めて表示する（縦横比保持・繰り返しなし）。
 *   - モード切替トグルは App.tsx のヘッダー右側ボタン群に統合された。
 */
import type { ChatMessage } from "../api";
import MessageList from "./MessageList";
import MessageInput from "./MessageInput";

interface Props {
  /** セッションID（入力下書きのキャッシュキーに使用） */
  sessionId: string;
  /** 表示するメッセージ一覧 */
  messages: ChatMessage[];
  /** キャラクター名（表示用） */
  characterName: string;
  /** ユーザ名（表示用） */
  userName: string;
  /** 送信処理中フラグ */
  sending: boolean;
  /** ストリーミング中のキャラクター応答テキスト（null = ストリーミングなし） */
  streamingContent: string | null;
  /** ストリーミング中の思考ブロック・想起記憶テキスト（null = なし） */
  streamingReasoning: string | null;
  /** 完了済みメッセージIDと reasoning テキストの対応マップ */
  reasoningMap: Record<string, string>;
  /**
   * メッセージ送信コールバック。
   * files には添付画像ファイルを渡す（空配列可）。
   */
  onSend: (content: string, files: File[]) => void;
  /**
   * ユーザメッセージ編集・キャラクター応答再生成コールバック。
   * fromMessageId 以降を削除して content で再送する。
   * imageIds には再送する画像IDリストを渡す（再生成時は元メッセージの画像を引き継ぐ）。
   */
  onRetry: (fromMessageId: string, content: string, imageIds: string[]) => void;
  /** スクロールに応じたヘッダー表示/非表示の通知コールバック。 */
  onHeaderVisibilityChange?: (visible: boolean) => void;
  /** char_msg_id → log_message_id のマッピング。バブルのログ折りたたみに使用する。 */
  msgLogIds?: Record<string, string>;
  /** char_msg_id → モデル応答完了までの経過時間（ミリ秒）のマッピング。 */
  elapsedMap?: Record<string, number>;
  /** 対面モードか（true なら背景・対面ブロック注入が有効）。 */
  faceToFaceMode?: boolean;
  /** 対面背景画像の URL（null/空なら背景なしで対面モードに入る）。 */
  faceToFaceBgUrl?: string | null;
}

/** 1on1チャットのレイアウトコンポーネント。 */
export default function ChatView({
  sessionId,
  messages,
  characterName,
  userName,
  sending,
  streamingContent,
  streamingReasoning,
  reasoningMap,
  onSend,
  onRetry,
  onHeaderVisibilityChange,
  msgLogIds,
  elapsedMap,
  faceToFaceMode = false,
  faceToFaceBgUrl = null,
}: Props) {
  // 対面モード + 背景画像があるときだけ background-image を当てる。
  // 画像未登録でも対面モード自体は有効（モードの意味は system prompt 注入が本体）。
  // contain: 画像の縦横比を保ったまま、画面に収まる最大サイズへ縮めて表示する。
  // 余白には ch-bg（テーマ色）を敷いて、画像のない領域を浮かせない。
  const wrapperStyle: React.CSSProperties = faceToFaceMode && faceToFaceBgUrl
    ? {
        backgroundImage: `url(${faceToFaceBgUrl})`,
        backgroundSize: "contain",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
        backgroundColor: "var(--ch-bg)",
      }
    : {};

  return (
    <div className="flex flex-col flex-1 h-full overflow-hidden relative" style={wrapperStyle}>
      <MessageList
        messages={messages}
        userName={userName}
        sending={sending}
        reasoningMap={reasoningMap}
        streamingContent={streamingContent}
        streamingReasoning={streamingReasoning}
        characterName={characterName}
        onRetry={onRetry}
        onHeaderVisibilityChange={onHeaderVisibilityChange}
        msgLogIds={msgLogIds}
        elapsedMap={elapsedMap}
        translucentBubbles={faceToFaceMode && !!faceToFaceBgUrl}
      />
      <MessageInput
        sessionId={sessionId}
        sending={sending}
        onSend={onSend}
        allowImages={true}
      />
    </div>
  );
}
