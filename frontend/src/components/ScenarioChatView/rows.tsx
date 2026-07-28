/**
 * シナリオチャットのバブル行 — ユーザ行・GM(NPC/narrator/character)行・あらすじ区切り。
 * UserBubbleRow / GMBubbleRow は React.memo 化されており、props が同一なら再描画しない。
 */
import React, { useEffect, useState } from "react";

import {
  Bubble,
  CharacterMessageRow,
  EditButton,
  MarkdownContent,
  MessageActionBar,
  ThinkingBlock,
  UserMessageActions,
  mobileBubbleExtendClass,
} from "../ChatBubbles";
import { trimEnd } from "./helpers";
import { Avatar } from "./npc";

interface UserBubbleRowProps {
  /** 表示する発話本体。再ストリーム後に親が turns を入れ替えると変わる。 */
  content: string;
  speaker_name: string;
  /** 編集アイコンを表示するか（送信中・終了セッション時は無効）。 */
  canEdit: boolean;
  /** 編集確定時のコールバック。新しい本文を受け取る。 */
  onCommit: (newContent: string) => void;
}

/**
 * ユーザ発話バブル。1on1 の UserBubble と同じ操作感:
 *   - ホバー時に右下に Copy ボタン + 鉛筆アイコン
 *   - 鉛筆クリックでインライン textarea に切り替わる
 *   - textarea で Ctrl+Enter で送信、Esc でキャンセル
 *
 * ユーザ発話はアバターを表示しない（右寄せのバブルのみ）。
 *
 * パフォーマンス: 末尾の `React.memo` でラップされる（同名定数を後段で再代入）。
 * 親から inline closure で渡る `onCommit` を比較対象から外し、
 * 表示に効くプリミティブ props だけで再レンダ可否を判定する。
 */
function UserBubbleRowImpl({
  content,
  speaker_name,
  canEdit,
  onCommit,
}: UserBubbleRowProps) {
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState(content);

  // content prop が変わった（再ストリーム完了で turns 再取得など）ら、
  // 編集テキストもサーバ最新値で初期化し直す。
  useEffect(() => {
    if (!editing) setEditText(content);
  }, [content, editing]);

  const submit = () => {
    const text = editText.trim();
    if (!text) return;
    setEditing(false);
    onCommit(text);
  };

  const cancel = () => {
    setEditing(false);
    setEditText(content);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // 下部の入力欄（MessageInput）と同じく Ctrl+Enter で送信する。
    if (e.key === "Enter" && e.ctrlKey) {
      e.preventDefault();
      submit();
    }
    if (e.key === "Escape") {
      cancel();
    }
  };

  return (
    // content-visibility: auto により、ビューポート外のバブルはレイアウト・ペイントが
    // スキップされる。長セッションで DOM が肥大した状態でも、入力欄でのタイピング・
    // スクロール時のフレーム落ちを軽減する目的。contain-intrinsic-size は概算プレース
    // ホルダーで、レンダリング後は実寸に置き換わる（スクロールバー長の予測精度向上）。
    <div
      className="flex justify-end group"
      style={{ contentVisibility: "auto", containIntrinsicSize: "auto 120px" }}
    >
      <div className="flex flex-col items-end max-w-full sm:max-w-[75%] w-full">
        <span className="text-[11px] text-ch-t4 mb-0.5 pr-1">{speaker_name}</span>
        {editing ? (
          <div className="flex flex-col gap-2 w-full">
            <textarea
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              onKeyDown={onKeyDown}
              rows={3}
              autoFocus
              className="rounded-xl px-4 py-2.5 text-sm resize-none focus:outline-none w-full"
              style={{ background: "rgb(var(--ch-ub))", color: "rgb(var(--ch-ut))" }}
            />
            <div className="flex gap-2 justify-end">
              <button
                onClick={cancel}
                className="text-ch-t3 hover:text-ch-t2 text-xs px-3 py-1.5 rounded transition-colors"
              >
                キャンセル
              </button>
              <button
                onClick={submit}
                disabled={!editText.trim()}
                className="text-white text-xs px-3 py-1.5 rounded font-medium transition-colors disabled:opacity-30"
                style={{ background: "var(--ch-accent)" }}
              >
                送信
              </button>
            </div>
            <p className="text-ch-t4 text-[10px] text-right">
              Ctrl+Enter で送信 / Esc でキャンセル（この発言以降は削除されます）
            </p>
          </div>
        ) : (
          <>
            {/* バブル本体（可変幅）。操作ボタンは下部に配置する。 */}
            <Bubble kind="user">
              <MarkdownContent content={trimEnd(content)} />
            </Bubble>
            <UserMessageActions
              copyText={content}
              onEdit={canEdit ? () => setEditing(true) : undefined}
            />
          </>
        )}
      </div>
    </div>
  );
}

interface GMBubbleRowProps {
  speaker_type: string;
  speaker_name: string;
  is_known: boolean | null;
  content: string;
  avatarSrc: string | null;
  /**
   * 自分が属する GM レスポンス（= 同一 raw_response の連続 GM バブル列 = 1 LLM 呼出ぶん）の最後尾なら true。
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
  const [editText, setEditText] = useState(content);

  // content prop がサーバ最新値に入れ替わったら編集テキストも初期化し直す。
  useEffect(() => {
    if (!editing) setEditText(content);
  }, [content, editing]);

  const submitEdit = () => {
    const text = editText.trim();
    if (!text) return;
    setEditing(false);
    onEditCommit?.(text);
  };

  const cancelEdit = () => {
    setEditing(false);
    setEditText(content);
  };

  const onEditKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && e.ctrlKey) {
      e.preventDefault();
      submitEdit();
    }
    if (e.key === "Escape") {
      cancelEdit();
    }
  };

  // 本文表示 or 編集フォーム。編集はバブル内で本文だけを差し替える
  // （ユーザ発話の編集と違い、この書き換えは先の展開に手を触れない）。
  const body = editing ? (
    <div className="flex flex-col gap-2 w-full">
      <textarea
        value={editText}
        onChange={(e) => setEditText(e.target.value)}
        onKeyDown={onEditKeyDown}
        rows={5}
        autoFocus
        className="rounded-lg px-3 py-2 text-sm resize-y focus:outline-none w-full"
        style={{ background: "rgb(var(--ch-s2))", color: "rgb(var(--ch-t1))" }}
      />
      <div className="flex gap-2 justify-end items-center">
        <p className="text-ch-t4 text-[10px] mr-auto">
          Ctrl+Enter で確定 / Esc でキャンセル（この応答の本文だけを書き換えます）
        </p>
        <button
          onClick={cancelEdit}
          className="text-ch-t3 hover:text-ch-t2 text-xs px-3 py-1.5 rounded transition-colors"
        >
          キャンセル
        </button>
        <button
          onClick={submitEdit}
          disabled={!editText.trim()}
          className="text-white text-xs px-3 py-1.5 rounded font-medium transition-colors disabled:opacity-30"
          style={{ background: "var(--ch-accent)" }}
        >
          確定
        </button>
      </div>
    </div>
  ) : (
    <MarkdownContent content={displayContent} />
  );

  // 編集の鉛筆はアバター列の下端（バブル左下の余白）へ置く。バブル下の操作バーへ足すと
  // 行が間延びするため。1 レスポンスは複数の話者ブロックに割れるので、末尾かどうかに
  // 関わらず全バブルに出す（末尾ブロックしか直せないと「無理やり直す」用途に届かない）。
  const editPencil =
    !editing && onEditCommit ? (
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
  // 行幅・スマホ時の左拡張は他モードと揃える（アバター列ぶんのスペーサーを置く）。
  // content-visibility: auto はビューポート外のレイアウト・ペイントをスキップ（DOM 肥大対策）。
  if (speaker_type === "narrator") {
    return (
      <div
        className="group flex gap-2.5 max-w-full sm:max-w-[88%]"
        style={{ contentVisibility: "auto", containIntrinsicSize: "auto 100px" }}
      >
        {/* アバター列ぶんのスペーサー。空きスペースの下端に編集の鉛筆を置く。 */}
        <div className="flex flex-col items-center" style={{ width: 28, flexShrink: 0 }}>
          {editPencil && <div className="mt-auto pb-1">{editPencil}</div>}
        </div>
        <div className="flex-1 min-w-0">
          {/* 鉛筆を置く行では列を空けたままにする（左拡張するとスペーサーごと覆ってしまう）。 */}
          <div className={editPencil ? "w-full" : mobileBubbleExtendClass}>
            <div
              className="text-sm leading-relaxed italic text-ch-t2 break-words"
              style={{ textWrap: "pretty" }}
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
    >
      {/* PC ターンの想起記憶・WM・思考ブロックを 1on1 と同じ ThinkingBlock で折りたたみ表示する。 */}
      {reasoning && <div className="mb-1"><ThinkingBlock content={reasoning} /></div>}
      <Bubble
        kind="character"
        colored
        characterName={speaker_name}
        dashed={is_known === false}
      >
        {body}
      </Bubble>
      {actions}
    </CharacterMessageRow>
  );
}

/**
 * UserBubbleRow / GMBubbleRow を `React.memo` でラップする。
 *
 * 親 (`ScenarioChatView`) は各バブルに inline closure で `onCommit` / `onRegenerate` /
 * `onAvatarClick` を渡しているため、デフォルトの浅い等価比較では関数 props の
 * 参照が毎レンダリングで変わって memo が無効化される。
 *
 * カスタム比較関数で「表示に効くプリミティブ props」だけを比較し、関数 props は
 * 無視する。これにより、ストリーミング中に最新バブルだけが変化しても他のバブルは
 * 再レンダリングされない（DOM 肥大時の入力もたつき・フレーム落ちを抑える）。
 *
 * 注: 関数 props が「捕捉している外部 state」が変わってもバブルは再描画されない。
 * 本プロジェクトでは関数の振る舞いはプリミティブ props（content / 等）と同期するため
 * 実用上問題ない（onCommit は turnId をクロージャーキャプチャするだけ、等）。
 */
export const UserBubbleRow = React.memo(UserBubbleRowImpl, (prev, next) => {
  return (
    prev.content === next.content &&
    prev.speaker_name === next.speaker_name &&
    prev.canEdit === next.canEdit
  );
});


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
    prev.reasoning === next.reasoning
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
