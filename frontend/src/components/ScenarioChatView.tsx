/**
 * シナリオチャットビュー。
 *
 * Ensemble エンジンが GM として全話者を演じるため、メッセージは ScenarioTurn として
 * `user | narrator | npc | character` の 4 種類の speaker_type を持つ。
 * 既存の ChatMessage / GroupChatView とは異なる表示が必要なため、自前の
 * バブルリストと入力欄を提供する。
 *
 * 機能:
 *   - 各バブルに `*xxx*` 斜体レンダリング（行動描写の独自記法）
 *   - Narrator は左寄せ・アバターなし・少し控えめなスタイル
 *   - User バブルクリックで編集 → 保存で「そのターン以降を削除＋再ストリーム」
 *   - 最終ユーザターンに「再生成」ボタン → 以降削除＋同内容で再ストリーム
 *   - ヘッダに Export ボタン（既存 ExportDialog を ScenarioTurn 用 adapter で流用）
 *
 * NPC の追加・編集はバックエンドの Scenarios UI で行う。
 */
import React, { useEffect, useMemo, useRef, useState } from "react";
import type {
  ScenarioSession,
  ScenarioTemplate,
  ScenarioNpc,
  ScenarioTurn,
} from "../api";
import {
  MarkdownContent,
  charHue,
  Bubble,
  CharacterMessageRow,
  MessageActionBar,
  UserMessageActions,
  mobileBubbleExtendClass,
} from "./ChatBubbles";
import MessageInput from "./MessageInput";
import { useHeaderVisibilityOnScroll } from "../hooks/useHeaderVisibilityOnScroll";

/** ストリーミング中の未確定吹き出し情報。
 *
 * `id` はクライアント側で `speaker_start` 受信時に発行する安定キー。
 * これを `key` に使うことで、`speaker_end` で配列から shift されたときに
 * インデックスがズレて他の pending バブルが再マウントされる事故を防ぐ。
 */
export interface PendingBubble {
  id: string;
  speaker_type: string;
  speaker_name: string;
  speaker_id: string | null;
  is_known: boolean;
  content: string;
}

interface Props {
  /** プレイセッション本体。 */
  session: ScenarioSession;
  /** 元のシナリオテンプレ（タイトル・場所などの表示に使う）。 */
  scenario: ScenarioTemplate | null;
  /** シナリオの NPC リスト（既知判定・アバター表示に使う）。 */
  npcs: ScenarioNpc[];
  /** これまでの確定ターン。 */
  turns: ScenarioTurn[];
  /** 送信中フラグ（true の間は入力欄無効化）。 */
  sending: boolean;
  /** ストリーミング中の未確定吹き出し列。 */
  pendingBubbles: PendingBubble[];
  /**
   * プレイヤー発話の送信。
   *
   * autoAdvance=true は「ユーザは無言で続きを促す」モード。
   * content は無視され、サーバ側で user turn を保存せず GM だけが発話する。
   */
  onSend: (content: string, autoAdvance?: boolean) => void;
  /** ユーザ発言の編集（fromTurn 以降を削除し、新しい content で再ストリーム）。 */
  onEditUserTurn: (turnId: string, newContent: string) => void;
  /** 最後のユーザターン以降を削除して同内容で再ストリーム。 */
  onRegenerate: () => void;
  /**
   * 末尾 GM ターン（同一 raw_response のバブル列）を削除する。
   * 再ストリームは行わず、ユーザリクエスト待ち状態へ戻す。
   * 主な用途: auto_advance で GM が応答した後、ユーザがその応答を捨てて
   * 自分の発話を入力したくなった場合。
   */
  onDiscard: () => void;
  /** スクロールに応じたヘッダー表示/非表示の通知コールバック。 */
  onHeaderVisibilityChange?: (visible: boolean) => void;
  /** turn_id → モデル応答完了までの経過時間（ミリ秒）のマッピング。 */
  elapsedMap?: Record<string, number>;
  /**
   * あらすじ作成バーの表示内容。null なら非表示。
   * `text` は「あらすじ未作成（X/Yターン）」等、`danger` が true（80%超）なら赤系で表示する。
   * 入力欄上部に常駐し、タップで作成モーダルを開く。
   */
  synopsisBar?: { text: string; danger: boolean } | null;
  /** 裏であらすじ蒸留が走っている最中か（控えめなインジケータ表示に使う）。 */
  synopsisGenerating?: boolean;
  /** あらすじ作成モーダルを開くコールバック（バーから呼ぶ）。 */
  onOpenSynopsisCreate?: () => void;
  /**
   * あらすじへ蒸留済みの最後の `turn_index`（`ScenarioSynopsis.last_turn_index`）。
   * 既定 -1（未蒸留）。これ以下の turn_index を持つバブルは「まとめ済み」、
   * これを超えるバブルは「まだまとめてない」として、境界に区切り線を描く。
   */
  synopsisLastTurnIndex?: number;
}

/** 文字列の末尾空白・改行を取り除く（表示時のノイズ除去）。 */
function trimEnd(s: string): string {
  return s.replace(/\s+$/u, "");
}

/** アバター用のプレースホルダー文字（名前の頭文字）。 */
function avatarInitial(name: string): string {
  return name?.[0] ?? "?";
}

/**
 * アバター画像 or プレースホルダーを表示する小さな円形要素。
 *
 * onClick が渡されると button としてラップされ、ホバーで微かに浮き上がる
 * （NPC 詳細ダイアログを開く操作のヒント）。onClick が無ければただの装飾。
 */
function Avatar(props: {
  name: string;
  src: string | null;
  size?: number;
  onClick?: () => void;
}) {
  const size = props.size ?? 28;
  const common = {
    width: size,
    height: size,
    borderRadius: "50%",
    flexShrink: 0,
  } as const;
  // 配色はキャラクター名から導出した色相を使い、1on1 チャットのアバターと揃える。
  const h = charHue(props.name);
  const inner = props.src ? (
    <img
      src={props.src}
      alt={props.name}
      style={{ ...common, objectFit: "cover" }}
      className="bg-ch-s2"
    />
  ) : (
    <div
      style={{
        ...common,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: size * 0.38,
        fontWeight: 600,
        background: `oklch(56% 0.12 ${h} / 0.15)`,
        color: `oklch(44% 0.14 ${h})`,
      }}
    >
      {avatarInitial(props.name)}
    </div>
  );

  if (!props.onClick) return inner;
  return (
    <button
      type="button"
      onClick={props.onClick}
      title={`${props.name} の詳細を表示`}
      className="rounded-full transition-transform hover:scale-105 hover:ring-1 hover:ring-ch-s3 focus:outline-none focus:ring-1 focus:ring-ch-accent"
      style={{ display: "inline-block", lineHeight: 0 }}
    >
      {inner}
    </button>
  );
}

/**
 * NPC 詳細ダイアログ。アバタークリックで開く。
 *
 * シンプル構成: 画像（大）/ 名前 / description（自由テキスト）。
 * 既知 NPC のみ呼ばれる想定（npc が null なら何も描画しない）。
 */
function NpcDetailDialog({
  npc,
  onClose,
}: {
  npc: ScenarioNpc | null;
  onClose: () => void;
}) {
  // Esc で閉じる挙動。1on1 等の他のダイアログと操作感を合わせる。
  useEffect(() => {
    if (!npc) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [npc, onClose]);

  if (!npc) return null;
  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center px-4"
      style={{ background: "var(--ch-overlay)" }}
      onClick={onClose}
    >
      <div
        className="bg-ch-bg rounded-xl w-full max-w-md overflow-hidden"
        style={{ border: "1px solid var(--ch-sep2)", boxShadow: "var(--ch-shadow)" }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* 画像エリア（あれば大きめ） */}
        {npc.image_data ? (
          <div className="w-full bg-ch-s1 flex items-center justify-center">
            <img
              src={npc.image_data}
              alt={npc.name}
              className="max-h-72 w-full object-contain"
            />
          </div>
        ) : (
          <div className="w-full bg-ch-s1 flex items-center justify-center py-12">
            <div
              style={{ width: 96, height: 96, borderRadius: "50%", fontSize: 40 }}
              className="bg-ch-s2 text-ch-t2 flex items-center justify-center"
            >
              {avatarInitial(npc.name)}
            </div>
          </div>
        )}

        {/* 本文 */}
        <div className="px-5 py-4">
          <div className="flex items-start justify-between gap-3 mb-3">
            <h2 className="text-ch-t1 text-base font-medium break-words">
              {npc.name}
            </h2>
            <button
              onClick={onClose}
              className="text-ch-t3 hover:text-ch-t1 text-sm -mr-1 -mt-1 p-1 rounded"
              aria-label="閉じる"
              title="閉じる (Esc)"
            >
              ✕
            </button>
          </div>
          {npc.description ? (
            <p className="text-ch-t2 text-sm leading-relaxed whitespace-pre-wrap break-words">
              {npc.description}
            </p>
          ) : (
            <p className="text-ch-t4 text-sm italic">説明はありません</p>
          )}
        </div>
      </div>
    </div>
  );
}

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
 *   - textarea で Shift+Enter で送信、Esc でキャンセル
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
    if (e.key === "Enter" && e.shiftKey) {
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
              Shift+Enter で送信 / Esc でキャンセル（この発言以降は削除されます）
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
  /** ターン末尾の GM バブルなら true。再生成 + Copy ボタンを表示する。 */
  isLastGM: boolean;
  /**
   * Copy ボタンがコピーするテキスト。末尾 GM バブルにのみ意味があり、
   * 「ターン全体（複数バブル）の本文連結」を渡すこと。
   */
  copyText?: string;
  /** 1ターンまるごと再生成（最終 user 以降を全削除して再ストリーム）。 */
  onRegenerate?: () => void;
  /** 末尾 GM ターンを破棄してユーザ入力待ちに戻す（再ストリームしない）。 */
  onDiscard?: () => void;
  /** アバタークリック時のコールバック。既知 NPC のみ渡される（押下可能になる）。 */
  onAvatarClick?: () => void;
  /** モデルへリクエストしてから応答完了までの経過時間（ミリ秒）。末尾 GM バブルにのみ意味がある。 */
  elapsedMs?: number;
}

/**
 * GM 側（Narrator / NPC / character）の発話バブル。1on1 の CharacterBubble に合わせて
 *   - 内容下に Copy ボタン
 *   - ターン末尾なら 1on1 と同じ ↺ アイコンで再生成（ホバー時に出現）
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
  isLastGM,
  copyText,
  onRegenerate,
  onDiscard,
  onAvatarClick,
  elapsedMs,
}: GMBubbleRowProps) {
  const displayContent = trimEnd(content);
  // 操作バー（コピー / 破棄 / 再生成）。ターン末尾の GM バブルにのみ表示する。
  // 1on1 / グループと共通の MessageActionBar を使う（DRY）。
  const actions = isLastGM ? (
    <MessageActionBar
      copyText={copyText ?? content}
      onRegenerate={onRegenerate}
      regenerateTitle="このターンを再生成"
      onDiscard={onDiscard}
      discardTitle="この応答を破棄してユーザ入力に戻す"
      elapsedMs={elapsedMs}
    />
  ) : null;

  // Narrator は地の文寄せ（アバターなし、見出しなし）。バブル枠を持たず斜体で流す。
  // 行幅・スマホ時の左拡張は他モードと揃える（アバター列ぶんのスペーサーを置く）。
  // content-visibility: auto はビューポート外のレイアウト・ペイントをスキップ（DOM 肥大対策）。
  if (speaker_type === "narrator") {
    return (
      <div
        className="group flex gap-2.5 max-w-full sm:max-w-[88%]"
        style={{ contentVisibility: "auto", containIntrinsicSize: "auto 100px" }}
      >
        <div style={{ width: 28, flexShrink: 0 }} />
        <div className="flex-1 min-w-0">
          <div className={mobileBubbleExtendClass}>
            <div
              className="text-sm leading-relaxed italic text-ch-t2 break-words"
              style={{ textWrap: "pretty" }}
            >
              <MarkdownContent content={displayContent} />
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
      nameSuffix={
        is_known === false ? (
          <span className="text-[10px] text-ch-t4">(ephemeral)</span>
        ) : undefined
      }
      style={{ contentVisibility: "auto", containIntrinsicSize: "auto 130px" }}
    >
      <Bubble
        kind="character"
        colored
        characterName={speaker_name}
        dashed={is_known === false}
      >
        <MarkdownContent content={displayContent} />
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
const UserBubbleRow = React.memo(UserBubbleRowImpl, (prev, next) => {
  return (
    prev.content === next.content &&
    prev.speaker_name === next.speaker_name &&
    prev.canEdit === next.canEdit
  );
});


const GMBubbleRow = React.memo(GMBubbleRowImpl, (prev, next) => {
  return (
    prev.speaker_type === next.speaker_type &&
    prev.speaker_name === next.speaker_name &&
    prev.is_known === next.is_known &&
    prev.content === next.content &&
    prev.avatarSrc === next.avatarSrc &&
    prev.isLastGM === next.isLastGM &&
    prev.copyText === next.copyText &&
    prev.elapsedMs === next.elapsedMs
  );
});

/**
 * あらすじまとめ済み／未まとめの境界を示す区切り線。
 *
 * 蒸留済みターン群（上）と未蒸留ターン群（下）の間に 1 本だけ挿入する。
 * 中央に「ここまであらすじにまとめ済み」ラベルを置き、左右に細い水平線を伸ばす。
 * 装飾的な目印なので `aria-hidden` とし、スクリーンリーダーには読み上げさせない。
 */
function SynopsisDivider() {
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

/** メイン: シナリオチャットビュー本体。 */
export default function ScenarioChatView({
  session,
  scenario,
  npcs,
  turns,
  sending,
  pendingBubbles,
  onSend,
  onEditUserTurn,
  onRegenerate,
  onDiscard,
  onHeaderVisibilityChange,
  elapsedMap,
  synopsisBar,
  synopsisGenerating,
  onOpenSynopsisCreate,
  synopsisLastTurnIndex,
}: Props) {
  /** クリックされた NPC の詳細ダイアログ表示用 state（null なら閉じている）。 */
  const [npcDialogTarget, setNpcDialogTarget] = useState<ScenarioNpc | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  /** スクロールに応じてヘッダー表示状態を判定する onScroll ハンドラ。 */
  const handleScroll = useHeaderVisibilityOnScroll(onHeaderVisibilityChange);

  /** 自動スクロール: turns / pendingBubbles が変わるたびに最下端へ追従。 */
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns, pendingBubbles]);

  /** NPC 名 → NPC オブジェクトのマップ（既知判定・アバター取得用）。 */
  const npcByName = useMemo(
    () => Object.fromEntries(npcs.map((n) => [n.name, n])),
    [npcs],
  );

  /**
   * あらすじ区切り線を直前に挿入すべきターンの id を計算する。
   *
   * `turn_index <= synopsisLastTurnIndex` がまとめ済み、それを超えるものが未まとめ。
   * その境界（=最初の未まとめターン）の id を返す。区切り線が無意味なケースは null:
   *   - synopsisLastTurnIndex 未指定 / -1（まだ何も蒸留していない）
   *   - まとめ済みターンが 1 つも無い（境界の上側が空 → 線を出す意味がない）
   *   - 未まとめターンが 1 つも無い（全ターン蒸留済み → 末尾に線が出て邪魔なだけ）
   */
  const synopsisDividerTurnId = useMemo(() => {
    const idx = synopsisLastTurnIndex ?? -1;
    if (idx < 0) return null;
    const hasSummarized = turns.some((t) => t.turn_index <= idx);
    if (!hasSummarized) return null;
    const firstUnsummarized = turns.find((t) => t.turn_index > idx);
    return firstUnsummarized ? firstUnsummarized.id : null;
  }, [turns, synopsisLastTurnIndex]);

  /**
   * 末尾 GM ターンのバブル一覧とそのまとめテキストを計算する。
   *
   * - lastGMTurnId: 末尾 GM 列の「最後のバブル」ID。再生成アイコン + Copy
   *   アイコンを表示する位置。
   * - lastGMTurnCopyText: その末尾 GM 列に属する全バブルの本文を `@名前: 本文`
   *   形式で連結したテキスト。Copy ボタンはターン全体（複数バブル分）をコピーする。
   *
   * 末尾 GM 列は raw_response で連続性を判定する: GM の 1 回の LLM 呼出で
   * 生成された一連の GM バブルが「同一ターン」を構成する。
   */
  const { lastGMTurnId, lastGMTurnCopyText } = useMemo(() => {
    if (turns.length === 0) {
      return { lastGMTurnId: null as string | null, lastGMTurnCopyText: "" };
    }
    let end = turns.length - 1;
    if (turns[end].speaker_type === "user") {
      return { lastGMTurnId: null as string | null, lastGMTurnCopyText: "" };
    }
    let start = end;
    const tailRaw = turns[end].raw_response;
    while (start > 0) {
      const prev = turns[start - 1];
      if (prev.speaker_type === "user") break;
      if (prev.raw_response !== tailRaw) break;
      start--;
    }
    const parts: string[] = [];
    for (let i = start; i <= end; i++) {
      const t = turns[i];
      const name =
        t.speaker_type === "narrator" ? "Narrator" : t.speaker_name;
      parts.push(`@${name}: ${trimEnd(t.content)}`);
    }
    return {
      lastGMTurnId: turns[end].id,
      lastGMTurnCopyText: parts.join("\n\n"),
    };
  }, [turns]);

  const resolveAvatar = (
    speaker_type: string,
    speaker_name: string,
  ): string | null => {
    if (speaker_type === "npc" || speaker_type === "character") {
      return npcByName[speaker_name]?.image_data ?? null;
    }
    return null;
  };

  const resolveIsKnown = (
    speaker_type: string,
    speaker_name: string,
  ): boolean | null => {
    if (speaker_type === "npc") return Boolean(npcByName[speaker_name]);
    if (speaker_type === "user" || speaker_type === "narrator") return true;
    return null;
  };

  /**
   * 共通 `MessageInput` からの送信を受ける。
   *
   * 入力が空の場合は「ユーザは無言で続きを促す」モード（auto_advance）で送る。
   * user turn は保存されず履歴に痕跡が残らない一方、GM プロンプトには
   * 「プレイヤーは今ターン何も発言していない」旨が OOC 指示として伝わる。
   * 画像添付はシナリオでは未対応のため第2引数（files）は無視する。
   */
  const handleScenarioInput = (content: string) => {
    const trimmed = content.trim();
    if (trimmed === "") {
      onSend("", true);
    } else {
      onSend(trimmed, false);
    }
  };

  const userPlaceholder = scenario?.user_alias ?? "プレイヤー";

  return (
    <div className="flex flex-col flex-1 h-full overflow-hidden">
      {/* チャットスクロール（1on1 と同じく最大幅 760px 中央寄せ・浮遊ヘッダー分の上余白） */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto" onScroll={handleScroll}>
       <div className="max-w-[760px] mx-auto px-4 sm:px-6 pt-16 pb-6 flex flex-col gap-5">
        {turns.length === 0 && pendingBubbles.length === 0 && (
          <div className="text-ch-t3 text-sm text-center mt-8">
            {scenario?.scenario ? (
              <pre className="whitespace-pre-wrap text-left mx-auto max-w-xl text-xs text-ch-t3 mb-4 italic">
                {scenario.scenario}
              </pre>
            ) : null}
            プレイヤー（{userPlaceholder}）として最初の発話を入力してシナリオを始めましょう
          </div>
        )}
        {turns.map((t) => {
          // このターンが「最初の未まとめターン」なら、直前にあらすじ区切り線を挿す。
          const divider =
            t.id === synopsisDividerTurnId ? <SynopsisDivider /> : null;
          let bubble: React.ReactNode;
          if (t.speaker_type === "user") {
            bubble = (
              <UserBubbleRow
                content={t.content}
                speaker_name={t.speaker_name}
                canEdit={session.status === "active" && !sending}
                onCommit={(newContent) => onEditUserTurn(t.id, newContent)}
              />
            );
          } else {
            // 既知 NPC のときだけアバタークリックを有効化（詳細ダイアログを開く）。
            // narrator や未知 NPC は対応する ScenarioNpc レコードが無いのでクリック無効。
            const npcForAvatar =
              t.speaker_type === "npc" || t.speaker_type === "character"
                ? npcByName[t.speaker_name] ?? null
                : null;
            bubble = (
              <GMBubbleRow
                speaker_type={t.speaker_type}
                speaker_name={t.speaker_name}
                is_known={resolveIsKnown(t.speaker_type, t.speaker_name)}
                content={t.content}
                avatarSrc={resolveAvatar(t.speaker_type, t.speaker_name)}
                isLastGM={
                  t.id === lastGMTurnId &&
                  session.status === "active" &&
                  !sending
                }
                copyText={lastGMTurnCopyText}
                onRegenerate={onRegenerate}
                onDiscard={onDiscard}
                onAvatarClick={
                  npcForAvatar ? () => setNpcDialogTarget(npcForAvatar) : undefined
                }
                elapsedMs={
                  t.id === lastGMTurnId ? elapsedMap?.[t.id] : undefined
                }
              />
            );
          }
          return (
            <React.Fragment key={t.id}>
              {divider}
              {bubble}
            </React.Fragment>
          );
        })}
        {pendingBubbles.map((b) => {
          const npcForAvatar =
            b.speaker_type === "npc" || b.speaker_type === "character"
              ? npcByName[b.speaker_name] ?? null
              : null;
          return (
            <GMBubbleRow
              key={b.id}
              speaker_type={b.speaker_type}
              speaker_name={b.speaker_name}
              is_known={b.is_known}
              content={b.content}
              avatarSrc={resolveAvatar(b.speaker_type, b.speaker_name)}
              isLastGM={false}
              onAvatarClick={
                npcForAvatar ? () => setNpcDialogTarget(npcForAvatar) : undefined
              }
            />
          );
        })}
        {sending && pendingBubbles.length === 0 && (
          <div className="text-ch-t3 text-xs italic self-start">
            GM が考えています…
          </div>
        )}
       </div>
      </div>

      {/* あらすじ作成中の控えめなインジケータ。生成は裏で走り、入力は継続できる。 */}
      {synopsisGenerating && (
        <div className="shrink-0 px-4 sm:px-6">
          <div className="max-w-[760px] mx-auto py-1.5 text-ch-t3 text-[11px] flex items-center gap-1.5">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-ch-t3 animate-pulse" />
            あらすじを作成中…
          </div>
        </div>
      )}

      {/* あらすじ作成バー。進捗が 50% を超えると表示（青）、80% 超で赤。
          テキストはターン側・文字側のうち限界に近い方を親が決めて渡す。
          生成中は上のインジケータに切り替わるため非表示。タップで作成モーダルを開く。 */}
      {synopsisBar && !synopsisGenerating && (
        <div className="shrink-0 px-4 sm:px-6">
          <button
            onClick={onOpenSynopsisCreate}
            className="w-full max-w-[760px] mx-auto mb-1.5 rounded-lg px-3 py-2 flex items-center justify-between gap-3 text-xs transition-opacity hover:opacity-90"
            style={{
              background: synopsisBar.danger
                ? "oklch(60% 0.18 25 / 0.12)"
                : "oklch(50% 0.13 226 / 0.10)",
              border: `1px solid ${
                synopsisBar.danger ? "oklch(60% 0.18 25 / 0.45)" : "var(--ch-sep2)"
              }`,
              color: synopsisBar.danger
                ? "oklch(55% 0.20 25)"
                : "rgb(var(--ch-t2))",
            }}
          >
            <span className="font-medium truncate">{synopsisBar.text}</span>
            <span className="shrink-0 text-[11px] opacity-80">あらすじ作成 ›</span>
          </button>
        </div>
      )}

      {/* 入力欄（1on1 / グループと共通の MessageInput を使う） */}
      {session.status === "active" ? (
        <MessageInput
          sessionId={session.id}
          sending={sending}
          onSend={handleScenarioInput}
          allowImages={false}
          allowEmptySend
          placeholder={`${userPlaceholder} として発話 (Ctrl+Enter で送信 / *手を握る* で行動描写 / 空欄送信で GM が無言のまま物語を進める)`}
        />
      ) : (
        <div
          className="shrink-0 px-3 py-3 text-ch-t3 text-sm text-center"
          style={{ borderTop: "1px solid var(--ch-sep)" }}
        >
          このセッションは終了しています
        </div>
      )}

      {/* NPC 詳細ダイアログ（アバタークリックで開く） */}
      <NpcDetailDialog
        npc={npcDialogTarget}
        onClose={() => setNpcDialogTarget(null)}
      />
    </div>
  );
}
