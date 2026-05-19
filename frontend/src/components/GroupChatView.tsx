/**
 * グループチャットビューコンポーネント。
 * MessageList・MessageInput を組み合わせてレイアウトするシェルコンポーネント。
 * 1on1チャットと同じ子コンポーネントを使い、グループ固有の props を受け渡す。
 *
 * ユーザターン時には、司会を介さず任意の参加者を手動指名できる操作バーを表示する。
 * 司会がエラーになった場合は、司会の再試行ボタンも併せて表示する。
 */
import type { ChatMessage } from "../api";
import MessageList from "./MessageList";
import MessageInput from "./MessageInput";
import { CharacterAvatar } from "./ChatBubbles";

interface Props {
  /** セッションID（入力下書きのキャッシュキーに使用） */
  sessionId: string;
  /** 表示するメッセージ一覧 */
  messages: ChatMessage[];
  /** グループ参加者のキャラクター名リスト（色割り当て順序・手動指名に使用） */
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
  /** スクロールに応じたヘッダー表示/非表示の通知コールバック。 */
  onHeaderVisibilityChange?: (visible: boolean) => void;
  /** ユーザターン待ち状態かどうか。true のときスキップ・手動指名操作を表示する。 */
  isUserTurn?: boolean;
  /** ユーザターンスキップコールバック。スキップボタン押下時に呼ばれる。 */
  onSkip?: () => void;
  /** 直近のターンで司会がエラーになったかどうか。true のとき司会再試行ボタンを表示する。 */
  directorErrored?: boolean;
  /** 司会の再試行コールバック。司会へもう一度次発言者の判断を依頼する。 */
  onRetryDirector?: () => void;
  /** 任意参加者の手動指名コールバック。司会を介さず指定キャラクターを発言させる。 */
  onRequestCharacter?: (charName: string) => void;
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
  isUserTurn = false,
  onSkip,
  directorErrored = false,
  onRetryDirector,
  onRequestCharacter,
}: Props) {
  // ユーザターン中かつ送信処理中でないときだけ手動操作バーを表示する。
  const showActionBar = isUserTurn && !sending;

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
      {showActionBar && (
        <div
          className="flex flex-col gap-1.5 px-3 py-2 shrink-0"
          style={{ borderTop: "1px solid var(--ch-sep)" }}
        >
          {/* 司会エラー時：再試行ボタン */}
          {directorErrored && (
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-amber-600">
                ⚠ 司会がエラーになりました
              </span>
              <button
                onClick={onRetryDirector}
                className="text-[11px] rounded-md px-2 py-0.5 transition-colors"
                style={{
                  border: "1px solid var(--ch-accent)",
                  color: "var(--ch-accent)",
                }}
              >
                司会を再試行
              </button>
            </div>
          )}
          {/* 手動指名：任意の参加者に直接発言を依頼する */}
          {participantNames.length > 0 && onRequestCharacter && (
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="text-[10px] text-ch-t3 font-mono shrink-0">
                発言を指名:
              </span>
              {participantNames.map((name) => (
                <button
                  key={name}
                  onClick={() => onRequestCharacter(name)}
                  className="flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] transition-colors"
                  style={{
                    border: "1px solid var(--ch-sep2)",
                    color: "rgb(var(--ch-t2))",
                  }}
                  title={`${name} に発言してもらう`}
                >
                  <CharacterAvatar characterName={name} size={16} />
                  {name}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      <MessageInput
        sessionId={sessionId}
        sending={sending}
        onSend={onSend}
        allowImages={true}
        onSkip={isUserTurn ? onSkip : undefined}
      />
    </div>
  );
}
