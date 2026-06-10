/**
 * メッセージ行コンポーネント — CharacterMessageRow / CharacterBubble / UserBubble。
 * 行レイアウト・バブル本体・操作バーを組み合わせた完成形の行部品。
 */
import React, { useState } from "react";

import { CharacterAvatar } from "./avatar";
import { Bubble } from "./Bubble";
import { UserMessageActions } from "./buttons";
import { ImageGrid } from "./images";
import { MarkdownContent } from "./markdown";
import { MessageActionBar } from "./MessageActionBar";
import { ThinkingBlock } from "./ThinkingBlock";

/**
 * バブル領域をスマホ幅で左へ広げ、左端をアバター左端に揃えるためのクラス。
 *
 * アバター幅(28px) + 余白(gap-2.5 = 10px) = 38px ぶん左へ拡張する。
 * sm 以上では従来どおりアバター右のインデント表示に戻す。
 * 1on1 / グループ / シナリオの全モードで共有する。
 */
export const mobileBubbleExtendClass =
  "-ml-[38px] w-[calc(100%_+_38px)] sm:ml-0 sm:w-full";

/**
 * キャラクター発話の共通行レイアウト（アバター + 名前行 + バブル領域）。
 *
 * 1on1 / グループ / シナリオ（NPC・character）でバブルの寸法・揃えを統一する
 * ための共通部品。各モードで重複していた flex 構造をここに集約している（DRY）。
 *
 * - アバターは常に上端揃え（`self-start`）。
 * - 行全体の最大幅は 88%（アバター込み）。
 * - `children`（バブル本体・操作バー等）はスマホ幅で左端をアバター左端に揃える。
 */
export function CharacterMessageRow({
  avatar,
  name,
  nameSuffix,
  style,
  testId,
  children,
}: {
  /** アバター要素（CharacterAvatar や NPC 用クリック可能アバター等）。 */
  avatar: React.ReactNode;
  /** 名前行に表示するキャラクター名。 */
  name: string;
  /** 名前の右に添える要素（@プリセット名・(ephemeral) ラベル等）。 */
  nameSuffix?: React.ReactNode;
  /** 行外側 div への追加 style（content-visibility 最適化用）。 */
  style?: React.CSSProperties;
  /** 行外側 div の data-testid。 */
  testId?: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className="group flex gap-2.5 max-w-full sm:max-w-[88%]"
      style={style}
      data-testid={testId}
    >
      {/* アバターは上端揃え（全モードで位置を統一）。 */}
      <div className="self-start">{avatar}</div>
      <div className="flex-1 min-w-0">
        {/*
         * 名前行: キャラクター名 + 補足。
         * スマホ幅では下のバブル領域を左へ広げる（左端をアバター左端に揃える）ため、
         * 名前行にアバターと同じ高さ(28px)を確保し、バブルがアバターへ重ならないようにする。
         */}
        <div className="flex items-center gap-1.5 flex-wrap mb-1 text-[11px] min-h-[28px] sm:min-h-0">
          <span className="font-semibold text-ch-t2">{name}</span>
          {nameSuffix}
        </div>
        {/* バブル領域: スマホ幅では左端をアバター左端に揃える */}
        <div className={mobileBubbleExtendClass}>{children}</div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// CharacterBubble
// ---------------------------------------------------------------------------

/**
 * キャラクターのチャットメッセージ（バブルスタイル）。
 * 小型アバター + 名前行（@プリセット）の右に、左上角を欠いた角丸バブルを表示する。
 * グループ/シナリオではキャラクター別の配色（cb0〜cb9）を、1on1 ではニュートラル面を使う。
 *
 * 行レイアウトは共通の CharacterMessageRow、操作バーは共通の MessageActionBar に委譲する。
 */
export function CharacterBubble({
  characterName,
  presetName,
  content,
  reasoning,
  colored = false,
  hue,
  sending = false,
  onRegenerate,
  logMessageId,
  elapsedMs,
}: {
  characterName: string;
  /** プリセット名。指定時は名前行に @プリセット を表示する。 */
  presetName?: string;
  content: string;
  reasoning?: string;
  /** true のときキャラクター別配色バブル（cb0〜cb9）を使う（グループ/シナリオ向け）。 */
  colored?: boolean;
  /** アバターの色相。省略時はキャラクター名から導出する。 */
  hue?: number;
  sending?: boolean;
  onRegenerate?: () => void;
  /** デバッグログフォルダ名（8桁hex）。存在する場合はログ折りたたみを表示する。 */
  logMessageId?: string;
  /** モデルへリクエストしてから応答完了までの経過時間（ミリ秒）。 */
  elapsedMs?: number;
}) {
  return (
    <CharacterMessageRow
      testId="character-bubble"
      // アバター画像は CharacterAvatar が CharacterImageContext から自動解決する。
      avatar={<CharacterAvatar characterName={characterName} hue={hue} size={28} />}
      name={characterName}
      nameSuffix={
        presetName ? (
          <span className="font-mono text-ch-t3 text-[0.95em]">@{presetName}</span>
        ) : undefined
      }
    >
      {reasoning && <div className="mb-1"><ThinkingBlock content={reasoning} /></div>}

      {/* バブル本体 */}
      <Bubble kind="character" colored={colored} characterName={characterName}>
        <MarkdownContent content={content} />
      </Bubble>

      {/* 操作バー（コピー / 経過時間 / ログ折りたたみ / 再生成）。logMessageId は CHOTGOR_DEBUG=1 時のみ。 */}
      {!sending && (
        <MessageActionBar
          copyText={content}
          onRegenerate={onRegenerate}
          logMessageId={logMessageId}
          elapsedMs={elapsedMs}
        />
      )}
    </CharacterMessageRow>
  );
}

// ---------------------------------------------------------------------------
// UserBubble
// ---------------------------------------------------------------------------

/**
 * ユーザーのチャットメッセージ。
 * 右寄せ・ニュートラルフラットデザイン。インライン編集フォームを内包する。
 */
export function UserBubble({
  content,
  userName,
  images,
  sending = false,
  onEdit,
}: {
  content: string;
  userName: string;
  images?: string[];
  sending?: boolean;
  onEdit?: (newContent: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState(content);

  const handleEditSubmit = () => {
    const text = editText.trim();
    if (!text) return;
    setEditing(false);
    onEdit?.(text);
  };

  const handleEditKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // 下部の入力欄（MessageInput）と同じく Ctrl+Enter で送信する。
    if (e.key === "Enter" && e.ctrlKey) {
      e.preventDefault();
      handleEditSubmit();
    }
    if (e.key === "Escape") {
      setEditing(false);
      setEditText(content);
    }
  };

  return (
    <div className="group flex flex-col items-end gap-0.5 max-w-full sm:max-w-[70%] ml-auto">
      {/* ユーザー名ラベル */}
      <span className="text-[11px] text-ch-t4 pr-1">{userName}</span>

      {/* 添付画像 */}
      {images && images.length > 0 && <ImageGrid imageIds={images} />}

      {editing ? (
        <div className="flex flex-col gap-2 w-full">
          <textarea
            value={editText}
            onChange={(e) => setEditText(e.target.value)}
            onKeyDown={handleEditKeyDown}
            rows={3}
            autoFocus
            className="rounded-xl px-4 py-2.5 text-sm resize-none focus:outline-none w-full"
            style={{ background: "rgb(var(--ch-ub))", color: "rgb(var(--ch-ut))" }}
          />
          <div className="flex gap-2 justify-end">
            <button
              onClick={() => { setEditing(false); setEditText(content); }}
              className="text-ch-t3 hover:text-ch-t2 text-xs px-3 py-1.5 rounded transition-colors"
            >
              キャンセル
            </button>
            <button
              onClick={handleEditSubmit}
              disabled={!editText.trim()}
              className="text-white text-xs px-3 py-1.5 rounded font-medium transition-colors disabled:opacity-30"
              style={{ background: "var(--ch-accent)" }}
            >
              送信
            </button>
          </div>
        </div>
      ) : (
        <>
          {/* バブル本体（インデント無し・可変幅）。操作ボタンは下部に配置する。 */}
          <Bubble kind="user">
            <MarkdownContent content={content} />
          </Bubble>
          {!sending && (
            <UserMessageActions
              copyText={content}
              onEdit={onEdit ? () => setEditing(true) : undefined}
            />
          )}
        </>
      )}
    </div>
  );
}
