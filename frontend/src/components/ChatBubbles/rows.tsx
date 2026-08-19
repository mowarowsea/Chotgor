/**
 * メッセージ行コンポーネント — CharacterMessageRow / CharacterBubble / UserBubble。
 * 行レイアウト・バブル本体・操作バーを組み合わせた完成形の行部品。
 */
import React, { useState } from "react";

import { useEditingLock } from "../../hooks/useEditingLock";
import { CharacterAvatar } from "./avatar";
import { Bubble } from "./Bubble";
import { UserMessageActions } from "./buttons";
import type { ActionHandler } from "./buttons";
import { ImageGrid } from "./images";
import { InlineEditor } from "./InlineEditor";
import { MarkdownContent } from "./markdown";
import { MessageActionBar } from "./MessageActionBar";
import { useRevealControls } from "./tapReveal";
import { ThinkingBlock } from "./ThinkingBlock";

/** ユーザ発話バブルの最大幅。キャラクター行（88%）より狭くして左右の非対称を保つ。 */
const userBubbleMaxWidthClass = "max-w-full sm:max-w-[70%]";

/**
 * キャラクター発話の共通行レイアウト（アバター + 名前行 + バブル領域）。
 *
 * 1on1 / グループ / シナリオ（NPC・character）でバブルの寸法・揃えを統一する
 * ための共通部品。各モードで重複していた flex 構造をここに集約している（DRY）。
 *
 * - アバターは常に列の上端。
 * - 行全体の最大幅は 88%（アバター込み）。
 * - アバター列は画面幅によらず確保する。列の下端は操作ボタン（鉛筆）の置き場所で、
 *   スマホ幅でバブルを左へ拡張すると、そこがバブルに覆われてしまうため。
 */
export function CharacterMessageRow({
  avatar,
  name,
  nameSuffix,
  underAvatar,
  style,
  testId,
  onPointerEnter,
  onPointerLeave,
  children,
}: {
  /** アバター要素（CharacterAvatar や NPC 用クリック可能アバター等）。 */
  avatar: React.ReactNode;
  /** 名前行に表示するキャラクター名。 */
  name: string;
  /** 名前の右に添える要素（@プリセット名・(ephemeral) ラベル等）。 */
  nameSuffix?: React.ReactNode;
  /** アバター列の下端（バブル左下の余白）へ置く要素。編集の鉛筆など。 */
  underAvatar?: React.ReactNode;
  /** 行外側 div への追加 style（content-visibility 最適化用）。 */
  style?: React.CSSProperties;
  /** 行外側 div の data-testid。 */
  testId?: string;
  /** 行のポインタ出入り。マウスホバーで操作ボタンを出す用途（呼び出し側で pointerType を見る）。 */
  onPointerEnter?: React.PointerEventHandler<HTMLDivElement>;
  onPointerLeave?: React.PointerEventHandler<HTMLDivElement>;
  children: React.ReactNode;
}) {
  return (
    <div
      className="group flex gap-2.5 max-w-full sm:max-w-[88%]"
      style={style}
      data-testid={testId}
      onPointerEnter={onPointerEnter}
      onPointerLeave={onPointerLeave}
    >
      {/* アバター列。アバターは上端（全モードで位置を統一）、underAvatar は下端へ落とす。
          underAvatar は絶対配置でフローから外す — 行が短いとアバターと縦に取り合って
          列の高さを押し広げ、ホバーのたびに行がガタつくため。短い行では
          アバターに重なるので z-10 で上に描く。 */}
      <div className="relative flex flex-col items-center shrink-0">
        {avatar}
        {underAvatar && (
          <div className="absolute bottom-0 left-1/2 -translate-x-1/2 z-10 rounded bg-ch-bg/80">
            {underAvatar}
          </div>
        )}
      </div>
      <div className="flex-1 min-w-0">
        {/* 名前行: キャラクター名 + 補足。アバター上端と揃う。 */}
        <div className="flex items-center gap-1.5 flex-wrap mb-1 text-[11px]">
          <span className="font-semibold text-ch-t2">{name}</span>
          {nameSuffix}
        </div>
        <div className="w-full">{children}</div>
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
 *
 * パフォーマンス: 末尾で `React.memo` 化される（同名定数を後段で再代入）。
 * 親から inline closure で渡る `onRegenerate` を比較対象から外し、表示に効く
 * プリミティブ props だけで再レンダ可否を判定する。
 */
function CharacterBubbleImpl({
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
  /** 再生成コールバック。Promise を返すと解決まで再生成ボタンが無効化される（二度押し防止）。 */
  onRegenerate?: ActionHandler;
  /** デバッグログフォルダ名（8桁hex）。存在する場合はログ折りたたみを表示する。 */
  logMessageId?: string;
  /** モデルへリクエストしてから応答完了までの経過時間（ミリ秒）。 */
  elapsedMs?: number;
}) {
  // 操作ボタンは露出中のバブルにだけ描画する（DOM 肥大対策 + 画面内で 1 つだけ）。
  const { revealed, rowProps, bubbleProps } = useRevealControls(!sending);

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
      // 画面外のレイアウト・ペイントをスキップする（長いセッションでの DOM 肥大対策）。
      style={{ contentVisibility: "auto", containIntrinsicSize: "auto 130px" }}
      {...rowProps}
    >
      {reasoning && <div className="mb-1"><ThinkingBlock content={reasoning} /></div>}

      {/* バブル本体 */}
      <Bubble
        kind="character"
        colored={colored}
        characterName={characterName}
        {...bubbleProps}
      >
        <MarkdownContent content={content} />
      </Bubble>

      {/* 操作バー（コピー / 経過時間 / ログ折りたたみ / 再生成）。logMessageId は CHOTGOR_DEBUG=1 時のみ。 */}
      {!sending && (
        <MessageActionBar
          copyText={content}
          onRegenerate={onRegenerate}
          logMessageId={logMessageId}
          elapsedMs={elapsedMs}
          revealed={revealed}
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
 *
 * パフォーマンス: 末尾で `React.memo` 化される（同名定数を後段で再代入）。
 * 親から inline closure で渡る `onEdit` は比較対象から外す。
 */
function UserBubbleImpl({
  content,
  userName,
  images,
  sending = false,
  onEdit,
  onDelete,
  editNote,
}: {
  content: string;
  userName: string;
  images?: string[];
  sending?: boolean;
  onEdit?: (newContent: string) => void;
  /**
   * 発言削除コールバック（無指定でゴミ箱非表示）。
   *
   * 後続の発話を巻き込まないよう、呼び出し側は「セッション末尾のユーザ発話」に
   * だけ渡すこと（キャラクター側の破棄ボタンと同じ制約）。
   */
  onDelete?: ActionHandler;
  /** 編集フォームのボタン行に添える注記（シナリオの「この発言以降は削除されます」等）。 */
  editNote?: string;
}) {
  const [editing, setEditing] = useState(false);
  // 編集中は下部の新規メッセージ入力・送信を無効化する。
  // 誤送信で以降のターンが巻き戻ると、編集中の内容ごと失われるため。
  useEditingLock(editing);
  // 操作ボタンは露出中のバブルにだけ描画する（DOM 肥大対策 + 画面内で 1 つだけ）。
  // ただし削除できるバブル（= 呼び出し側が末尾と判定して onDelete を渡したもの）は
  // ホバーを待たずに出す。破棄・削除の直後はホバーの反映が一拍遅れるため。
  const { revealed, rowProps, bubbleProps } = useRevealControls(
    !sending && !editing,
    onDelete !== undefined,
  );

  return (
    <div
      className={`group flex flex-col items-end gap-0.5 ml-auto ${userBubbleMaxWidthClass}`}
      // 画面外のレイアウト・ペイントをスキップする（長いセッションでの DOM 肥大対策）。
      style={{ contentVisibility: "auto", containIntrinsicSize: "auto 120px" }}
      {...rowProps}
    >
      {/* ユーザー名ラベル */}
      <span className="text-[11px] text-ch-t4 pr-1">{userName}</span>

      {/* 添付画像 */}
      {images && images.length > 0 && <ImageGrid imageIds={images} />}

      {editing ? (
        <InlineEditor
          value={content}
          tone="user"
          note={editNote}
          onSubmit={(newContent) => {
            setEditing(false);
            onEdit?.(newContent);
          }}
          onCancel={() => setEditing(false)}
        />
      ) : (
        <>
          {/* バブル本体（インデント無し・可変幅）。操作ボタンは下部に配置する。 */}
          <Bubble kind="user" {...bubbleProps}>
            <MarkdownContent content={content} />
          </Bubble>
          {!sending && (
            <UserMessageActions
              copyText={content}
              onEdit={onEdit ? () => setEditing(true) : undefined}
              onDelete={onDelete}
              revealed={revealed}
            />
          )}
        </>
      )}
    </div>
  );
}

/**
 * CharacterBubble / UserBubble を `React.memo` でラップする。
 *
 * 親（MessageList）は各バブルへ inline closure でコールバックを渡すため、既定の
 * 浅い比較では関数 props の参照が毎レンダリングで変わって memo が無効になる。
 * 表示に効くプリミティブ props だけを比較し、関数 props は無視する
 * （シナリオの GMBubbleRow / UserBubbleRow と同じ方針）。
 *
 * これにより、ストリーミング中に最新バブルだけが変化しても既存バブルは
 * 再レンダリングされない。
 */
export const CharacterBubble = React.memo(CharacterBubbleImpl, (prev, next) => {
  return (
    prev.characterName === next.characterName &&
    prev.presetName === next.presetName &&
    prev.content === next.content &&
    prev.reasoning === next.reasoning &&
    prev.colored === next.colored &&
    prev.hue === next.hue &&
    prev.sending === next.sending &&
    prev.logMessageId === next.logMessageId &&
    prev.elapsedMs === next.elapsedMs &&
    // 関数の同一性は見ないが「渡されているか」は見る（操作の可否が切り替わるため）。
    (prev.onRegenerate === undefined) === (next.onRegenerate === undefined)
  );
});

export const UserBubble = React.memo(UserBubbleImpl, (prev, next) => {
  return (
    prev.content === next.content &&
    prev.userName === next.userName &&
    prev.sending === next.sending &&
    prev.editNote === next.editNote &&
    // 関数の同一性は見ないが「渡されているか」は見る（編集・削除の可否が切り替わるため）。
    // onDelete は末尾バブルでのみ渡るので、新しい発話が積まれた時点でゴミ箱を引っ込める。
    (prev.onEdit === undefined) === (next.onEdit === undefined) &&
    (prev.onDelete === undefined) === (next.onDelete === undefined) &&
    // 画像は ID の並びで比較する（親が毎回新しい配列を渡しても参照差で落ちないように）。
    (prev.images ?? []).join(",") === (next.images ?? []).join(",")
  );
});
