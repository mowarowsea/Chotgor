/**
 * 1on1チャットビューコンポーネント。
 * MessageList・MessageInput を組み合わせてレイアウトするだけのシェルコンポーネント。
 * メッセージ描画・入力フォームのロジックはそれぞれの子コンポーネントに委譲する。
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
  /** スクロール方向変化コールバック。MessageList から App へ伝播する。 */
  onHeaderVisibilityChange?: (visible: boolean) => void;
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
}: Props) {
  return (
    <div className="flex flex-col flex-1 h-full overflow-hidden">
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
