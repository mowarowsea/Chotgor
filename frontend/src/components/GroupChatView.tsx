/**
 * グループチャットビューコンポーネント。
 * MessageList・MessageInput を組み合わせてレイアウトするだけのシェルコンポーネント。
 * 1on1チャットと同じ子コンポーネントを使い、グループ固有の props を受け渡す。
 */
import type { ChatMessage } from "../api";
import MessageList from "./MessageList";
import MessageInput from "./MessageInput";

interface Props {
  /** セッションID（入力下書きのキャッシュキーに使用） */
  sessionId: string;
  /** 表示するメッセージ一覧 */
  messages: ChatMessage[];
  /** グループ参加者のキャラクター名リスト（色割り当て順序に使用） */
  participantNames: string[];
  /** ユーザ名（表示用） */
  userName: string;
  /** 送信処理中フラグ */
  sending: boolean;
  /** 応答待機中・ストリーミング中のキャラクター名（null = なし） */
  waitingCharacter: string | null;
  /** ストリーミング中の応答テキスト */
  streamingContent?: string | null;
  /** ストリーミング中の思考ブロック・想起記憶テキスト */
  streamingReasoning?: string | null;
  /** メッセージIDに紐付いた thinking/reasoning テキスト */
  reasoningMap: Record<string, string>;
  /**
   * メッセージ送信コールバック。
   * 1on1チャットと同じシグネチャ（files には添付画像ファイルを渡す）。
   */
  onSend: (content: string, files: File[]) => void;
  /**
   * ユーザメッセージ編集・キャラクター応答再生成コールバック。
   * fromMessageId 以降を削除して content で再送する。
   * 1on1チャットと同じシグネチャ（imageIds には再送する画像IDリスト）。
   */
  onRetry?: (fromMessageId: string, content: string, imageIds: string[]) => void;
  /** スクロール方向変化コールバック。MessageList から App へ伝播する。 */
  onHeaderVisibilityChange?: (visible: boolean) => void;
}

/** グループチャットのレイアウトコンポーネント。 */
export default function GroupChatView({
  sessionId,
  messages,
  participantNames,
  userName,
  sending,
  waitingCharacter,
  streamingContent = null,
  streamingReasoning = null,
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
        participantNames={participantNames}
        waitingCharacter={waitingCharacter}
        streamingContent={streamingContent}
        streamingReasoning={streamingReasoning}
        emptyMessage="グループチャットを始めましょう"
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
