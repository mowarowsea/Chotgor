/**
 * シナリオチャットのバブル行 — GM(NPC/narrator/character)行・あらすじ区切り。
 * ユーザ発話は 1on1 と共通の UserBubble を使う（ScenarioChatView から直接）。
 * GMBubbleRow は React.memo 化されており、props が同一なら再描画しない。
 */
import React, { useState } from "react";

import {
  Bubble,
  CharacterMessageRow,
  EditButton,
  InlineEditor,
  MarkdownContent,
  MessageActionBar,
  ThinkingBlock,
  useRevealControls,
} from "../ChatBubbles";
import { trimEnd } from "./helpers";
import { Avatar } from "./npc";

/**
 * ナレーター（地の文）の行頭マーク。アバターを持たない地の文に、
 * 話者ブロックの始まりを示す目印を与える。長さを変えた三本線で「文章の行」を表す。
 * 装飾でしかないので aria-hidden とし、読み上げ対象から外す。
 */
function NarratorMark() {
  return (
    <svg
      width={14}
      height={14}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      className="text-ch-t4"
      aria-hidden
    >
      <path d="M4 7h16M4 12h16M4 17h10" />
    </svg>
  );
}

interface GMBubbleRowProps {
  speaker_type: string;
  speaker_name: string;
  is_known: boolean | null;
  content: string;
  avatarSrc: string | null;
  /**
   * 自分が属する GM レスポンス（= 同一 response_key の連続 GM バブル列 = 1 LLM 呼出ぶん）の最後尾なら true。
   * グループ末尾だけが MessageActionBar（コピー + ログ + 必要なら再生成）を持つ。
   */
  isGroupTail: boolean;
  /**
   * 「最新グループの末尾」なら true。再生成 + 破棄ボタンを有効にする条件。
   * `isGroupTail` の真部分集合（最新グループ末尾は当然グループ末尾でもある）。
   */
  isLastGM: boolean;
  /**
   * Copy ボタンがコピーするテキスト。グループ末尾バブルでのみ参照される。
   * 自グループ全バブルを `@名前: 本文` 形式で連結した文字列を渡すこと。
   */
  copyText?: string;
  /** 1 レスポンスまるごと再生成（最終 user 以降を巻き戻して再ストリーム）。 */
  onRegenerate?: () => void;
  /** 末尾 GM レスポンスを破棄してユーザ入力待ちに戻す（再ストリームしない）。 */
  onDiscard?: () => void;
  /** 発話の手動書き換え確定。新しい本文を受け取る。 */
  onEditCommit?: (newContent: string) => void;
  /** 枝ナビの現在位置と総数（このバブルが属するレスポンスの兄弟枝）。 */
  variantIndex?: number;
  variantCount?: number;
  /** 前後の枝へ切り替える。過去レスポンスでは呼び出し側が確認を挟む。 */
  onPrevVariant?: () => void;
  onNextVariant?: () => void;
  /** アバタークリック時のコールバック。既知 NPC のみ渡される（押下可能になる）。 */
  onAvatarClick?: () => void;
  /** モデルへリクエストしてから応答完了までの経過時間（ミリ秒）。最新グループ末尾でのみ意味がある。 */
  elapsedMs?: number;
  /** デバッグログフォルダ名（8桁 hex）。指定時のみ ▼ログ 折りたたみを表示する。 */
  logMessageId?: string;
  /** 想起記憶・WM・思考ブロック（pc_reasoning の連結）。PC ターンでのみ非空になる。
   *  指定時はバブル先頭に折りたたみ ThinkingBlock を表示し、1on1 のキャラクター応答と同じ見た目にする。 */
  reasoning?: string;
}

/**
 * GM 側（Narrator / NPC / character）の発話バブル。1on1 の CharacterBubble に合わせて
 *   - 内容下に Copy ボタン
 *   - レスポンス末尾なら 1on1 と同じ ↺ アイコンで再生成（ホバー時に出現）
 * を配置する。
 *
 * パフォーマンス: 末尾の `React.memo` でラップされる（同名定数を後段で再代入）。
 * `onRegenerate` `onAvatarClick` は親から inline closure で渡るため比較対象から外し、
 * 表示に効くプリミティブ props だけで再レンダ可否を判定する。
 * 結果として、ストリーミング中に変化していない既存バブルは再描画されない。
 */
function GMBubbleRowImpl({
  speaker_type,
  speaker_name,
  is_known,
  content,
  avatarSrc,
  isGroupTail,
  isLastGM,
  copyText,
  onRegenerate,
  onDiscard,
  onEditCommit,
  variantIndex,
  variantCount,
  onPrevVariant,
  onNextVariant,
  onAvatarClick,
  elapsedMs,
  logMessageId,
  reasoning,
}: GMBubbleRowProps) {
  const displayContent = trimEnd(content);
  const [editing, setEditing] = useState(false);
  // 操作ボタンは露出中のバブルにだけ描画する（DOM 肥大対策 + 画面内で 1 つだけ）。
  const { revealed, rowProps, bubbleProps } = useRevealControls(!editing);

  // 本文表示 or 編集フォーム。編集はバブル内で本文だけを差し替える
  // （ユーザ発話の編集と違い、この書き換えは先の展開に手を触れない）。
  const body = editing ? (
    <InlineEditor
      value={content}
      tone="character"
      rows={5}
      note="Ctrl+Enter で確定 / Esc でキャンセル（この応答の本文だけを書き換えます）"
      submitLabel="確定"
      onSubmit={(newContent) => {
        setEditing(false);
        onEditCommit?.(newContent);
      }}
      onCancel={() => setEditing(false)}
    />
  ) : (
    <MarkdownContent content={displayContent} />
  );

  // 編集の鉛筆はアバター列の下端（バブル左下の余白）へ置く。バブル下の操作バーへ足すと
  // 行が間延びするため。1 レスポンスは複数の話者ブロックに割れるので、末尾かどうかに
  // 関わらず全バブルに出す（末尾ブロックしか直せないと「無理やり直す」用途に届かない）。
  const editPencil =
    !editing && onEditCommit && revealed ? (
      <EditButton onClick={() => setEditing(true)} title="この発話を書き換える" />
    ) : undefined;

  // 操作バー（コピー / 枝ナビ / 破棄 / 再生成 / ログ）。
  // 1on1 / グループと共通の MessageActionBar を使う（DRY）。
  // モデル応答は複数バブルで構成され得るため、グループ末尾のバブルにだけ操作バーを出して
  // 1 応答 = 1 操作バーを保つ。再生成・破棄は最新グループ末尾でのみ有効。
  // 編集はここに置かない（鉛筆はアバター列の下端へ出す）。
  const actions =
    editing || !isGroupTail ? null : (
      <MessageActionBar
        copyText={copyText ?? content}
        onRegenerate={isLastGM ? onRegenerate : undefined}
        regenerateTitle="このレスポンスを再生成（前の結果は枝として残る）"
        onDiscard={isLastGM ? onDiscard : undefined}
        discardTitle="この応答を破棄してユーザ入力に戻す"
        variantIndex={variantIndex}
        variantCount={variantCount}
        onPrevVariant={onPrevVariant}
        onNextVariant={onNextVariant}
        elapsedMs={elapsedMs}
        logMessageId={logMessageId}
      />
    );

  // Narrator は地の文寄せ（アバターなし、見出しなし）。バブル枠を持たず斜体で流す。
  // 左のスペーサー列は鉛筆の置き場所として常に確保する（スマホ幅でも左拡張しない）。
  // 鉛筆の有無で幅を変えると、ストリーミング中〜確定でレイアウトががたつくため。
  // content-visibility: auto はビューポート外のレイアウト・ペイントをスキップ（DOM 肥大対策）。
  if (speaker_type === "narrator") {
    return (
      <div
        className="group flex gap-2.5 max-w-full sm:max-w-[88%]"
        style={{ contentVisibility: "auto", containIntrinsicSize: "auto 100px" }}
        {...rowProps}
      >
        {/* アバター列ぶんのスペーサー。上端にナレーターマーク、空きスペースの下端に編集の鉛筆を置く。
            マークの mt は本文 1 行目（text-sm / leading-relaxed）の中心に合わせた値。
            鉛筆は絶対配置でフローから外す（CharacterMessageRow と同じ理由 — 短い行で
            列の高さを押し広げてガタつかせないため）。 */}
        <div className="relative flex flex-col items-center" style={{ width: 28, flexShrink: 0 }}>
          <div className="mt-[5px]">
            <NarratorMark />
          </div>
          {editPencil && (
            <div className="absolute bottom-0 left-1/2 -translate-x-1/2 z-10 rounded bg-ch-bg/80">
              {editPencil}
            </div>
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="w-full">
            <div
              className="text-sm leading-relaxed italic text-ch-t2 break-words"
              style={{ textWrap: "pretty" }}
              {...bubbleProps}
            >
              {body}
            </div>
            {actions}
          </div>
        </div>
      </div>
    );
  }

  // NPC / character: 左寄せ + アバター。共通の CharacterMessageRow に委譲する。
  // content-visibility: auto はビューポート外のレイアウト・ペイントをスキップ（DOM 肥大対策）。
  return (
    <CharacterMessageRow
      avatar={<Avatar name={speaker_name} src={avatarSrc} onClick={onAvatarClick} />}
      name={speaker_name}
      underAvatar={editPencil}
      nameSuffix={
        is_known === false ? (
          <span className="text-[10px] text-ch-t4">(ephemeral)</span>
        ) : undefined
      }
      style={{ contentVisibility: "auto", containIntrinsicSize: "auto 130px" }}
      {...rowProps}
    >
      {/* PC ターンの想起記憶・WM・思考ブロックを 1on1 と同じ ThinkingBlock で折りたたみ表示する。 */}
      {reasoning && <div className="mb-1"><ThinkingBlock content={reasoning} /></div>}
      <Bubble
        kind="character"
        colored
        characterName={speaker_name}
        dashed={is_known === false}
        {...bubbleProps}
      >
        {body}
      </Bubble>
      {actions}
    </CharacterMessageRow>
  );
}

/**
 * GMBubbleRow を `React.memo` でラップする。
 *
 * 親 (`ScenarioChatView`) は各バブルに inline closure で `onRegenerate` /
 * `onAvatarClick` を渡しているため、デフォルトの浅い等価比較では関数 props の
 * 参照が毎レンダリングで変わって memo が無効化される。
 *
 * カスタム比較関数で「表示に効くプリミティブ props」だけを比較し、関数 props は
 * 無視する。これにより、ストリーミング中に最新バブルだけが変化しても他のバブルは
 * 再レンダリングされない（DOM 肥大時の入力もたつき・フレーム落ちを抑える）。
 *
 * 注: 関数 props が「捕捉している外部 state」が変わってもバブルは再描画されない。
 * 本プロジェクトでは関数の振る舞いはプリミティブ props（content / 等）と同期するため
 * 実用上問題ない（onEditCommit は turnId をクロージャーキャプチャするだけ、等）。
 */
export const GMBubbleRow = React.memo(GMBubbleRowImpl, (prev, next) => {
  return (
    prev.speaker_type === next.speaker_type &&
    prev.speaker_name === next.speaker_name &&
    prev.is_known === next.is_known &&
    prev.content === next.content &&
    prev.avatarSrc === next.avatarSrc &&
    prev.isGroupTail === next.isGroupTail &&
    prev.isLastGM === next.isLastGM &&
    prev.copyText === next.copyText &&
    prev.variantIndex === next.variantIndex &&
    prev.variantCount === next.variantCount &&
    prev.elapsedMs === next.elapsedMs &&
    prev.logMessageId === next.logMessageId &&
    prev.reasoning === next.reasoning &&
    // 関数の同一性は見ないが「渡されているか」は見る（書き換えの可否が切り替わるため）。
    (prev.onEditCommit === undefined) === (next.onEditCommit === undefined)
  );
});

/**
 * あらすじまとめ済み／未まとめの境界を示す区切り線。
 *
 * 蒸留済みターン群（上）と未蒸留ターン群（下）の間に 1 本だけ挿入する。
 * 中央に「ここまであらすじにまとめ済み」ラベルを置き、左右に細い水平線を伸ばす。
 * 装飾的な目印なので `aria-hidden` とし、スクリーンリーダーには読み上げさせない。
 */
export function SynopsisDivider() {
  return (
    <div className="flex items-center gap-3 -my-1 select-none" aria-hidden>
      <div className="flex-1 h-px" style={{ background: "var(--ch-sep2)" }} />
      <span className="text-[10px] text-ch-t4 flex items-center gap-1 whitespace-nowrap">
        <svg
          width="11"
          height="11"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.4"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M20 6 9 17l-5-5" />
        </svg>
        ここまであらすじにまとめ済み
      </span>
      <div className="flex-1 h-px" style={{ background: "var(--ch-sep2)" }} />
    </div>
  );
}
