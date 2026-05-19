/**
 * シナリオチャット（Zeta モード）ビュー。
 *
 * Ensemble エンジンが GM として全話者を演じるため、メッセージは ZetaTurn として
 * `user | narrator | npc | character` の 4 種類の speaker_type を持つ。
 * 既存の ChatMessage / GroupChatView とは異なる表示が必要なため、自前の
 * バブルリストと入力欄を提供する。
 *
 * 機能:
 *   - 各バブルに `*xxx*` 斜体レンダリング（行動描写の独自記法）
 *   - Narrator は左寄せ・アバターなし・少し控えめなスタイル
 *   - User バブルクリックで編集 → 保存で「そのターン以降を削除＋再ストリーム」
 *   - 最終ユーザターンに「再生成」ボタン → 以降削除＋同内容で再ストリーム
 *   - ヘッダに Export ボタン（既存 ExportDialog を ZetaTurn 用 adapter で流用）
 *
 * NPC の追加・編集はバックエンドの Scenarios UI で行う。
 */
import React, { useEffect, useMemo, useRef, useState } from "react";
import type {
  ScenarioSession,
  ScenarioSynopsis,
  ScenarioTemplate,
  ZetaNpc,
  ZetaTurn,
} from "../api";
import { CopyButton, MarkdownContent, charHue, Bubble } from "./ChatBubbles";
import MessageInput from "./MessageInput";

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
  npcs: ZetaNpc[];
  /** これまでの確定ターン。 */
  turns: ZetaTurn[];
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
  /** セッションのあらすじ（記憶捏造対策）。未取得は null。 */
  synopsis: ScenarioSynopsis | null;
  /** あらすじを部分更新（auto / manual のどちらか / 両方）。 */
  onSynopsisChange: (patch: { auto?: string; manual?: string }) => Promise<void>;
  /** synopsis_auto への自動追記フローを手動起動する。 */
  onSynopsisRegenerate: () => Promise<void>;
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
  npc: ZetaNpc | null;
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

/**
 * 鉛筆アイコン（編集ボタン）。1on1 の UserBubble と同じ svg を流用する。
 * UI の一貫性のため、別コンポーネントながら見た目を完全に揃えている。
 */
function PencilIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      strokeWidth={1.5}
      stroke="currentColor"
      width={12}
      height={12}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L10.582 16.07a4.5 4.5 0 0 1-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 0 1 1.13-1.897l8.932-8.931Zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0 1 15.75 21H5.25A2.25 2.25 0 0 1 3 18.75V8.25A2.25 2.25 0 0 1 5.25 6H10"
      />
    </svg>
  );
}

interface UserBubbleRowProps {
  /** 表示する発話本体。再ストリーム後に親が turns を入れ替えると変わる。 */
  content: string;
  speaker_name: string;
  avatarSrc: string | null;
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
 * シナリオ固有の差分はアバター（右側に小さい円形）のみ。
 *
 * パフォーマンス: 末尾の `React.memo` でラップされる（同名定数を後段で再代入）。
 * 親から inline closure で渡る `onCommit` を比較対象から外し、
 * 表示に効くプリミティブ props だけで再レンダ可否を判定する。
 */
function UserBubbleRowImpl({
  content,
  speaker_name,
  avatarSrc,
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
      className="flex justify-end gap-2.5 group"
      style={{ contentVisibility: "auto", containIntrinsicSize: "auto 120px" }}
    >
      <div className="flex flex-col items-end max-w-[70%] w-full">
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
          <div className="flex items-end gap-1 flex-row-reverse min-w-0">
            <Bubble kind="user">
              <MarkdownContent content={trimEnd(content)} />
            </Bubble>
            <div className="flex items-center gap-0.5">
              <CopyButton text={content} />
              {canEdit && (
                <button
                  onClick={() => setEditing(true)}
                  title="編集"
                  className="opacity-100 sm:opacity-0 sm:group-hover:opacity-100 text-ch-t3 hover:text-ch-t2 transition-all p-1 rounded shrink-0"
                >
                  <PencilIcon />
                </button>
              )}
            </div>
          </div>
        )}
      </div>
      <Avatar name={speaker_name} src={avatarSrc} />
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
  /** アバタークリック時のコールバック。既知 NPC のみ渡される（押下可能になる）。 */
  onAvatarClick?: () => void;
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
  onAvatarClick,
}: GMBubbleRowProps) {
  const displayContent = trimEnd(content);
  const actions = (
    <BubbleActions
      copyText={copyText ?? content}
      isLastGM={isLastGM}
      onRegenerate={onRegenerate}
    />
  );

  // Narrator は地の文寄せ（アバターなし、見出しなし）。
  // content-visibility: auto はビューポート外のレイアウト・ペイントをスキップ（DOM 肥大対策）。
  if (speaker_type === "narrator") {
    // Narrator は地の文寄せ（アバターなし、見出しなし）。バブル枠を持たず斜体で流す。
    return (
      <div
        className="flex justify-start gap-2.5 group"
        style={{ contentVisibility: "auto", containIntrinsicSize: "auto 100px" }}
      >
        <div style={{ width: 28, flexShrink: 0 }} />
        <div className="flex flex-col items-start max-w-[88%]">
          <div
            className="text-sm leading-relaxed italic text-ch-t2 break-words"
            style={{ textWrap: "pretty" }}
          >
            <MarkdownContent content={displayContent} />
          </div>
          {actions}
        </div>
      </div>
    );
  }

  // NPC / character: 左寄せ + アバター
  // content-visibility: auto はビューポート外のレイアウト・ペイントをスキップ（DOM 肥大対策）。
  return (
    <div
      className="flex justify-start gap-2.5 group"
      style={{ contentVisibility: "auto", containIntrinsicSize: "auto 130px" }}
    >
      <Avatar name={speaker_name} src={avatarSrc} onClick={onAvatarClick} />
      <div className="flex flex-col items-start max-w-[88%] min-w-0">
        <div className="flex items-center gap-1.5 flex-wrap mb-1 text-[11px]">
          <span className="font-semibold text-ch-t2">{speaker_name}</span>
          {is_known === false && (
            <span className="text-[10px] text-ch-t4">(ephemeral)</span>
          )}
        </div>
        <Bubble
          kind="character"
          colored
          characterName={speaker_name}
          dashed={is_known === false}
        >
          <MarkdownContent content={displayContent} />
        </Bubble>
        {actions}
      </div>
    </div>
  );
}

/**
 * GM バブル下に並ぶ操作アイコン列。
 * Copy と ↺ 再生成は「ターンの末尾バブル」に対してのみ表示する。
 * Copy はターン全体（複数バブル分）の本文をまとめてコピーするため、
 * 末尾バブル単体のテキストではなく親が計算した copyText を渡している。
 */
function BubbleActions({
  copyText,
  isLastGM,
  onRegenerate,
}: {
  copyText: string;
  isLastGM: boolean;
  onRegenerate?: () => void;
}) {
  if (!isLastGM) return null;
  return (
    <div className="flex items-center gap-0.5 -ml-1 mt-0.5">
      <CopyButton text={copyText} />
      {onRegenerate && (
        <button
          onClick={onRegenerate}
          title="このターンを再生成"
          className="opacity-100 sm:opacity-0 sm:group-hover:opacity-100 text-ch-t3 hover:text-ch-t2 text-xs transition-all p-1 rounded"
        >
          ↺
        </button>
      )}
    </div>
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
    prev.avatarSrc === next.avatarSrc &&
    prev.canEdit === next.canEdit
  );
});

/**
 * あらすじパネル — セッション単位の自動要約 (auto) と手動補足 (manual) の編集 UI。
 *
 * 記憶捏造対策として導入:
 *   - synopsis_auto: LLM が古い経緯を自動で要約・追記したもの。GM プロンプトのメイン。
 *     ユーザは UI からここを自由編集できる（捏造記述を発見したら削除・修正できる）。
 *   - synopsis_manual: プレイヤーが手で書き留めた補足メモ。自動更新で破壊されない。
 *     GM への補正指示として機能する（auto と矛盾するときは manual が優先される）。
 *
 * 折りたたみ式で、デフォルトは閉じている（ターン履歴を遮らないため）。
 */
function SynopsisPanel({
  synopsis,
  onChange,
  onRegenerate,
  disabled,
}: {
  synopsis: ScenarioSynopsis | null;
  onChange: (patch: { auto?: string; manual?: string }) => Promise<void>;
  onRegenerate: () => Promise<void>;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [autoDraft, setAutoDraft] = useState("");
  const [manualDraft, setManualDraft] = useState("");
  const [savingAuto, setSavingAuto] = useState(false);
  const [savingManual, setSavingManual] = useState(false);
  const [regenerating, setRegenerating] = useState(false);

  // synopsis（サーバ最新値）が変わったら、未編集の draft を同期する。
  // 編集中の値はパネルを閉じない限り保持される（誤って巻き戻されないため）。
  useEffect(() => {
    if (!open) {
      setAutoDraft(synopsis?.auto ?? "");
      setManualDraft(synopsis?.manual ?? "");
    }
  }, [synopsis, open]);

  const toggleOpen = () => {
    if (!open) {
      // 開く瞬間にサーバ最新値で draft を初期化（前回の draft が残らないように）
      setAutoDraft(synopsis?.auto ?? "");
      setManualDraft(synopsis?.manual ?? "");
    }
    setOpen(!open);
  };

  const saveAuto = async () => {
    if (savingAuto) return;
    setSavingAuto(true);
    try {
      await onChange({ auto: autoDraft });
    } finally {
      setSavingAuto(false);
    }
  };

  const saveManual = async () => {
    if (savingManual) return;
    setSavingManual(true);
    try {
      await onChange({ manual: manualDraft });
    } finally {
      setSavingManual(false);
    }
  };

  const regenerate = async () => {
    if (regenerating) return;
    setRegenerating(true);
    try {
      await onRegenerate();
    } finally {
      setRegenerating(false);
    }
  };

  const autoLen = (synopsis?.auto ?? "").length;
  const manualLen = (synopsis?.manual ?? "").length;
  const summary = open
    ? ""
    : autoLen + manualLen === 0
      ? "（未生成）"
      : `自動 ${autoLen} 文字 / 補足 ${manualLen} 文字`;

  return (
    <div
      className="shrink-0 px-3 py-2 bg-ch-bg"
      style={{ borderBottom: "1px solid var(--ch-sep)" }}
    >
      <button
        onClick={toggleOpen}
        className="text-ch-t2 hover:text-ch-t1 text-xs flex items-center gap-2 w-full"
      >
        <span>{open ? "▼" : "▶"}</span>
        <span className="font-medium">これまでのあらすじ</span>
        <span className="text-ch-t4 ml-auto">{summary}</span>
      </button>
      {open && (
        <div className="mt-2 flex flex-col gap-3">
          {/* auto: メインのあらすじ（LLM 自動生成・追記） */}
          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between text-[11px] text-ch-t3">
              <span>自動あらすじ（メイン。LLM が古い履歴を要約・追記）</span>
              <div className="flex gap-2">
                <button
                  onClick={regenerate}
                  disabled={disabled || regenerating}
                  className="text-ch-t3 hover:text-ch-t1 text-[11px] px-2 py-0.5 rounded disabled:opacity-30"
                  style={{ border: "1px solid var(--ch-sep2)" }}
                  title="今すぐ古い履歴を要約して追記する"
                >
                  {regenerating ? "生成中…" : "追記更新"}
                </button>
                <button
                  onClick={saveAuto}
                  disabled={disabled || savingAuto || autoDraft === (synopsis?.auto ?? "")}
                  className="text-ch-accent-t bg-ch-accent-dim text-[11px] px-2 py-0.5 rounded disabled:opacity-30"
                  style={{ border: "1px solid oklch(50% 0.13 226 / 0.30)" }}
                  title="編集内容を保存（捏造記述を削除・修正するのに使う）"
                >
                  {savingAuto ? "保存中…" : "保存"}
                </button>
              </div>
            </div>
            <textarea
              value={autoDraft}
              onChange={(e) => setAutoDraft(e.target.value)}
              rows={6}
              placeholder="（履歴が上限を超えると LLM が自動で要約・追記します）"
              className="bg-ch-s1 text-ch-t1 rounded px-3 py-2 text-sm resize-y focus:outline-none"
              style={{ border: "1px solid var(--ch-sep2)" }}
            />
          </div>
          {/* manual: プレイヤー手書きの補足メモ */}
          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between text-[11px] text-ch-t3">
              <span>
                補足メモ（手書き。自動更新では破壊されない・GM への補正指示）
              </span>
              <button
                onClick={saveManual}
                disabled={disabled || savingManual || manualDraft === (synopsis?.manual ?? "")}
                className="text-ch-accent-t bg-ch-accent-dim text-[11px] px-2 py-0.5 rounded disabled:opacity-30"
                style={{ border: "1px solid oklch(50% 0.13 226 / 0.30)" }}
              >
                {savingManual ? "保存中…" : "保存"}
              </button>
            </div>
            <textarea
              value={manualDraft}
              onChange={(e) => setManualDraft(e.target.value)}
              rows={4}
              placeholder="例: 主人公はレイカと「絶対に裏切らない」と約束した。"
              className="bg-ch-s1 text-ch-t1 rounded px-3 py-2 text-sm resize-y focus:outline-none"
              style={{ border: "1px solid var(--ch-sep2)" }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

const GMBubbleRow = React.memo(GMBubbleRowImpl, (prev, next) => {
  return (
    prev.speaker_type === next.speaker_type &&
    prev.speaker_name === next.speaker_name &&
    prev.is_known === next.is_known &&
    prev.content === next.content &&
    prev.avatarSrc === next.avatarSrc &&
    prev.isLastGM === next.isLastGM &&
    prev.copyText === next.copyText
  );
});

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
  synopsis,
  onSynopsisChange,
  onSynopsisRegenerate,
}: Props) {
  /** クリックされた NPC の詳細ダイアログ表示用 state（null なら閉じている）。 */
  const [npcDialogTarget, setNpcDialogTarget] = useState<ZetaNpc | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

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
      {/* あらすじパネル（記憶捏造対策・折りたたみ式） */}
      <SynopsisPanel
        synopsis={synopsis}
        onChange={onSynopsisChange}
        onRegenerate={onSynopsisRegenerate}
        disabled={sending}
      />
      {/* チャットスクロール（1on1 と同じく最大幅 760px 中央寄せ） */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
       <div className="max-w-[760px] mx-auto px-4 sm:px-6 py-6 flex flex-col gap-5">
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
          if (t.speaker_type === "user") {
            return (
              <UserBubbleRow
                key={t.id}
                content={t.content}
                speaker_name={t.speaker_name}
                avatarSrc={resolveAvatar(t.speaker_type, t.speaker_name)}
                canEdit={session.status === "active" && !sending}
                onCommit={(newContent) => onEditUserTurn(t.id, newContent)}
              />
            );
          }
          // 既知 NPC のときだけアバタークリックを有効化（詳細ダイアログを開く）。
          // narrator や未知 NPC は対応する ZetaNpc レコードが無いのでクリック無効。
          const npcForAvatar =
            t.speaker_type === "npc" || t.speaker_type === "character"
              ? npcByName[t.speaker_name] ?? null
              : null;
          return (
            <GMBubbleRow
              key={t.id}
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
              onAvatarClick={
                npcForAvatar ? () => setNpcDialogTarget(npcForAvatar) : undefined
              }
            />
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
