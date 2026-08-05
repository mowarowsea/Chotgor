/**
 * シナリオチャットビュー。
 *
 * Ensemble エンジンが GM として全話者を演じるため、メッセージは ScenarioTurn として
 * `user | narrator | npc | character` の 4 種類の speaker_type を持つ。
 * 既存の ChatMessage（1on1）とは異なる表示が必要なため、自前の
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
import React, {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import type {
  ScenarioNpc,
  ScenarioSession,
  ScenarioTemplate,
  ScenarioTurn,
} from "../../api";
import { EditingLockProvider, useEditingLocked } from "../../hooks/useEditingLock";
import { useHeaderVisibilityOnScroll } from "../../hooks/useHeaderVisibilityOnScroll";
import { CharacterAvatar, ThinkingBlock, UserBubble } from "../ChatBubbles";
import MessageInput from "../MessageInput";
import { trimEnd } from "./helpers";
import { NpcDetailDialog } from "./npc";
import { GMBubbleRow, SynopsisDivider } from "./rows";

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
  /** これまでの確定ターン（直近ウィンドウぶん。過去は遡り読み込みで伸びる）。 */
  turns: ScenarioTurn[];
  /** 表示中ウィンドウより過去のターンが残っているか（遡りボタンの表示可否）。 */
  hasOlderTurns?: boolean;
  /** 遡り読み込みの実行中フラグ。 */
  loadingOlderTurns?: boolean;
  /** 表示ウィンドウを 1 ページぶん過去へ伸ばす。 */
  onLoadOlder?: () => void;
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
   * 末尾 GM レスポンス（同一 response_key のバブル列 = 1 LLM 呼出ぶん）を削除する。
   * 再ストリームは行わず、ユーザリクエスト待ち状態へ戻す。
   * 主な用途: auto_advance で GM が応答した後、ユーザがその応答を捨てて
   * 自分の発話を入力したくなった場合。
   */
  onDiscard: () => void;
  /** 枝（レスポンスガチャ）の切替。分岐点より後の本線は巻き戻される。 */
  onSwitchVariant: (generationId: string) => void;
  /** GM / PC 発話の手動書き換え（枝は生やさず本文だけ差し替える）。 */
  onEditResponse: (turnId: string, newContent: string) => void;
  /**
   * ensemble_pc 専用「ターンを譲る」操作。指定先（PC枠名 / "GM" / "ALL"）に発話を回す。
   * チップ群が押されたときに呼ばれる。バックエンドでは auto_advance=true + yield_to で処理され、
   * ユーザ発話は履歴に残らない。
   */
  onYieldTo?: (target: string) => void;
  /** スクロールに応じたヘッダー表示/非表示の通知コールバック。 */
  onHeaderVisibilityChange?: (visible: boolean) => void;
  /** turn_id → モデル応答完了までの経過時間（ミリ秒）のマッピング。 */
  elapsedMap?: Record<string, number>;
  /** 生成中レスポンスの reasoning（想起記憶・WM・スケッチ）。1on1 の streamingReasoning と同じ役割で、
   *  未確定バブルの頭にライブ表示する。確定ターンぶんは `ScenarioTurn.reasoning` が持つ。 */
  scenarioStreamingReasoning?: string | null;
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

/** メイン: シナリオチャットビュー本体（編集ロックの Provider 内で描画される）。 */
function ScenarioChatViewInner({
  session,
  scenario,
  npcs,
  turns,
  hasOlderTurns = false,
  loadingOlderTurns = false,
  onLoadOlder,
  sending,
  pendingBubbles,
  onSend,
  onEditUserTurn,
  onRegenerate,
  onDiscard,
  onSwitchVariant,
  onEditResponse,
  onYieldTo,
  onHeaderVisibilityChange,
  elapsedMap,
  scenarioStreamingReasoning,
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
  /**
   * ユーザ発話バブルが編集中か。編集中は「発話の送信」に類する操作をすべて止める。
   * 入力欄本体は MessageInput が自分で無効化するので、ここでは入力欄の外にある
   * 宛先トグル・「ターンを譲る」チップ群を抑える。
   */
  const editingLocked = useEditingLocked();

  /**
   * 自動スクロール: 末尾ターンが変わったとき / ストリーミング中に最下端へ追従。
   *
   * 依存を「末尾ターンの id」にしているのは、遡り読み込み（先頭への追加）で
   * 最下端へ飛ばされないようにするため。前方に足しても末尾は変わらない。
   */
  const lastTurnId = turns.length > 0 ? turns[turns.length - 1].id : null;
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lastTurnId, pendingBubbles]);

  /**
   * 遡り読み込みで先頭に追加されたぶん、見ている位置がずれないよう補正する。
   *
   * 発火時に「下端からの距離（scrollHeight - scrollTop）」を控えておき、
   * DOM 反映直後に同じ距離へ戻す。描画前に補正したいので useLayoutEffect を使う。
   */
  const olderScrollAnchorRef = useRef<number | null>(null);
  useLayoutEffect(() => {
    const el = scrollRef.current;
    const anchor = olderScrollAnchorRef.current;
    if (el && anchor !== null) {
      el.scrollTop = el.scrollHeight - anchor;
      olderScrollAnchorRef.current = null;
    }
  }, [turns]);

  /**
   * 無限スクロール: 履歴の上端に近づいたら過去を 1 ページぶん自動で取りに行く。
   *
   * `loadingOlderTurns` を依存に入れているのは、読み込み後もセンチネルが可視域に
   * 残っている（1 ページでは埋まらなかった）ときに再購読で再発火させるため。
   * IntersectionObserver は交差状態が「変化した」ときしか呼ばれないので、
   * 購読し直さないと 2 ページ目以降が出ない。
   */
  const topSentinelRef = useRef<HTMLDivElement>(null);
  /** 親から渡る関数は毎レンダー変わりうるので、購読を張り直さずに済むよう ref 経由で呼ぶ。 */
  const onLoadOlderRef = useRef(onLoadOlder);
  onLoadOlderRef.current = onLoadOlder;
  useEffect(() => {
    if (!hasOlderTurns || loadingOlderTurns) return;
    const root = scrollRef.current;
    const target = topSentinelRef.current;
    if (!root || !target) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (!entries[0]?.isIntersecting) return;
        olderScrollAnchorRef.current = root.scrollHeight - root.scrollTop;
        onLoadOlderRef.current?.();
      },
      // 上端に届く前から先読みして、遡り続けても途切れて見えないようにする
      { root, rootMargin: "600px 0px 0px 0px" },
    );
    io.observe(target);
    return () => io.disconnect();
  }, [hasOlderTurns, loadingOlderTurns]);

  /** NPC 名 → NPC オブジェクトのマップ（既知判定・アバター取得用）。 */
  const npcByName = useMemo(
    () => Object.fromEntries(npcs.map((n) => [n.name, n])),
    [npcs],
  );

  /** PC枠名 → PcSlot のマップ（PC アバター取得用）。 */
  const pcSlotByName = useMemo(
    () => Object.fromEntries((scenario?.pc_slots ?? []).map((s) => [s.name, s])),
    [scenario],
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
   * GM ターン（=話者ブロック）を「同一モデル応答」単位にグルーピングし、各グループ末尾のバブルに
   * 関する情報（ID と全バブル連結テキスト）を返す。各グループが「1 レスポンス」に対応する。
   *
   * - gmGroupTailById: グループ末尾の turn id → そのグループ全バブルを連結した
   *   コピー用テキスト。MessageActionBar（コピー/ログ/再生成）はグループ末尾のみに出す。
   * - lastGMTurnId: 「最新グループ」の末尾バブル id。再生成 / 破棄の対象となるのは
   *   このバブルだけ（過去グループはコピー + ログのみ）。末尾が user / turns が空なら null。
   *
   * グルーピングは response_key（raw_response の指紋）の連続性で判定する: GM の 1 回の
   * LLM 呼出で生成された一連の GM 話者ブロックが「同一レスポンス」を構成する。
   * user ターンは境界。
   */
  const { gmGroupTailById, lastGMTurnId } = useMemo(() => {
    const tailById = new Map<string, string>();
    if (turns.length === 0) {
      return { gmGroupTailById: tailById, lastGMTurnId: null as string | null };
    }
    // 連続する同 response_key の GM ターン（=話者ブロック）を 1 レスポンスグループとして畳む。
    // 末尾に達するか次が user / response_key 不一致になったらフラッシュする。
    const flushGroup = (start: number, end: number) => {
      const parts: string[] = [];
      for (let i = start; i <= end; i++) {
        const t = turns[i];
        const name =
          t.speaker_type === "narrator" ? "Narrator" : t.speaker_name;
        parts.push(`@${name}: ${trimEnd(t.content)}`);
      }
      tailById.set(turns[end].id, parts.join("\n\n"));
    };
    let groupStart = -1;
    let groupKey: string | null | undefined = undefined;
    for (let i = 0; i < turns.length; i++) {
      const t = turns[i];
      if (t.speaker_type === "user") {
        if (groupStart >= 0) flushGroup(groupStart, i - 1);
        groupStart = -1;
        groupKey = undefined;
        continue;
      }
      // GM ターン（=話者ブロック）: 既存レスポンスグループと response_key が一致しなければ新レスポンス
      if (groupStart < 0 || t.response_key !== groupKey) {
        if (groupStart >= 0) flushGroup(groupStart, i - 1);
        groupStart = i;
        groupKey = t.response_key;
      }
    }
    if (groupStart >= 0) flushGroup(groupStart, turns.length - 1);

    // 最新グループ末尾は turns の末尾が GM である場合のみ存在
    const lastTurn = turns[turns.length - 1];
    const lastGMId = lastTurn.speaker_type === "user" ? null : lastTurn.id;
    return { gmGroupTailById: tailById, lastGMTurnId: lastGMId };
  }, [turns]);

  const resolveAvatar = (
    speaker_type: string,
    speaker_name: string,
  ): string | null => {
    if (speaker_type === "npc" || speaker_type === "character") {
      return npcByName[speaker_name]?.image_data ?? null;
    }
    // pc はシナリオの pc_slots に登録されたアバター（image_data）を名前で解決する
    if (speaker_type === "pc") {
      return pcSlotByName[speaker_name]?.image_data ?? null;
    }
    return null;
  };

  const resolveIsKnown = (
    speaker_type: string,
    speaker_name: string,
  ): boolean | null => {
    if (speaker_type === "npc") return Boolean(npcByName[speaker_name]);
    if (speaker_type === "user" || speaker_type === "narrator") return true;
    // pc は配役確定済み（PC 起動時にバリデーション済み）なので known 扱い
    if (speaker_type === "pc") return true;
    return null;
  };

  /**
   * 宛先 PC トグルの状態。null=OFF（メンションなし）、それ以外は @<name> として
   * 送信メッセージ先頭に付与する文字列。ユーザがメッセージを送るたびに OFF に戻す。
   * クリックで OFF → PC1 → PC2 → … → "ALL" → OFF を循環する。
   */
  const [mentionTarget, setMentionTarget] = useState<string | null>(null);

  /**
   * 共通 `MessageInput` からの送信を受ける。
   *
   * 入力が空の場合は「ユーザは無言で続きを促す」モード（auto_advance）で送る。
   * user turn は保存されず履歴に痕跡が残らない一方、GM プロンプトには
   * 「プレイヤーは今回何も発言していない」旨が OOC 指示として伝わる。
   * 画像添付はシナリオでは未対応のため第2引数（files）は無視する。
   * 宛先 PC トグルが ON の場合は本文先頭に `@<target> ` を付ける。
   */
  const handleScenarioInput = (content: string) => {
    const trimmed = content.trim();
    if (trimmed === "") {
      onSend("", true);
    } else {
      const prefix = mentionTarget ? `@${mentionTarget} ` : "";
      onSend(prefix + trimmed, false);
    }
    // 送信のたびに宛先トグルを OFF へ戻す（ユーザターンが来るたびにリセット）。
    setMentionTarget(null);
  };

  const userPlaceholder = scenario?.pc_slots?.[0]?.name ?? "プレイヤー";

  // ensemble_pc（TRPG モード）では @<PC枠名> で PC を指名できることを入力欄でヒントする。
  // session.pc_assignments には slot_id しか入っていないので、scenario.pc_slots から name を引く。
  const isPcMode = session.engine_type === "ensemble_pc";
  const pcSlotsById = new Map(
    (scenario?.pc_slots ?? []).map((s) => [s.slot_id, s]),
  );
  const pcSlotNames = (session.pc_assignments ?? [])
    .filter((pc) => pc.player_type === "character")
    .map((pc) => pcSlotsById.get(pc.slot_id)?.name)
    .filter((n): n is string => !!n);
  const inputPlaceholder = `${userPlaceholder} として発話`;

  /**
   * 宛先トグルの候補（PCモードのみ）。AI キャラ枠の名前 + "ALL" を循環する。
   * ユーザ枠は自分自身宛になるため候補から除外する。
   */
  const mentionOptions =
    isPcMode && pcSlotNames.length > 0 ? [...pcSlotNames, "ALL"] : [];

  /** クリックで宛先を OFF → PC1 → PC2 → … → ALL → OFF と循環させる。 */
  const cycleMentionTarget = () => {
    if (mentionOptions.length === 0) return;
    setMentionTarget((prev) => {
      if (prev === null) return mentionOptions[0];
      const idx = mentionOptions.indexOf(prev);
      if (idx === -1 || idx === mentionOptions.length - 1) return null;
      return mentionOptions[idx + 1];
    });
  };

  /** 宛先ボタン: MessageInput の extraTools として送信ボタンの直前に描画される。
   *  ON のときはアクセント色 + `@<name>` 表示、OFF は `@` アイコンのみ。 */
  const mentionToggleButton =
    mentionOptions.length > 0 ? (
      <button
        type="button"
        onClick={cycleMentionTarget}
        disabled={sending || editingLocked}
        title={
          mentionTarget
            ? `宛先: @${mentionTarget}（クリックで切替）`
            : "宛先 PC を指定（クリックで切替）"
        }
        className="text-xs rounded px-1.5 py-0.5 transition-colors disabled:opacity-30 max-w-[120px] truncate"
        style={{
          color: mentionTarget ? "var(--ch-accent)" : "var(--ch-t3)",
          border: `1px solid ${
            mentionTarget ? "var(--ch-accent)" : "var(--ch-sep2)"
          }`,
          lineHeight: 1.2,
        }}
      >
        {mentionTarget ? `@${mentionTarget}` : "@"}
      </button>
    ) : null;

  return (
    <div className="flex flex-col flex-1 h-full overflow-hidden">
      {/* チャットスクロール（1on1 と同じく最大幅 760px 中央寄せ・浮遊ヘッダー分の上余白） */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto" onScroll={handleScroll}>
       <div className="max-w-[760px] mx-auto px-4 sm:px-6 pt-16 pb-6 flex flex-col gap-5">
        {/* 遡り読み込みのセンチネル。可視域に入ると 1 ページぶん過去へ自動で伸びる。 */}
        {hasOlderTurns && (
          <div
            ref={topSentinelRef}
            className="self-center text-ch-t4 text-[11px] min-h-[20px] flex items-center"
          >
            {loadingOlderTurns && <span className="animate-pulse">読み込み中...</span>}
          </div>
        )}
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
              // ユーザ発話バブルは 1on1 と共通（右寄せ・編集フォーム・操作バー）。
              // 編集確定で以降のターンが消えるため、注記でそれを伝える。
              <UserBubble
                content={t.content}
                userName={t.speaker_name}
                editNote="Ctrl+Enter で送信 / Esc でキャンセル（この発言以降は削除されます）"
                onEdit={
                  session.status === "active" && !sending
                    ? (newContent) => onEditUserTurn(t.id, newContent)
                    : undefined
                }
              />
            );
          } else {
            // 既知 NPC のときだけアバタークリックを有効化（詳細ダイアログを開く）。
            // narrator や未知 NPC は対応する ScenarioNpc レコードが無いのでクリック無効。
            const npcForAvatar =
              t.speaker_type === "npc" || t.speaker_type === "character"
                ? npcByName[t.speaker_name] ?? null
                : null;
            const groupCopyText = gmGroupTailById.get(t.id);
            const isGroupTail = groupCopyText !== undefined;
            const isLastGM =
              t.id === lastGMTurnId &&
              session.status === "active" &&
              !sending;
            const canMutate = session.status === "active" && !sending;
            const variantCount = t.variant_count ?? 1;
            const variantIndex = t.variant_index ?? 1;
            const siblings = t.variant_siblings ?? [];
            // 枝ナビは「本線の末尾レスポンス以外」だと下流の巻き戻しを伴うため確認を挟む。
            // 巻き戻した続きは復元しない（docs/planned/scenario_turn_variants_plan.md）。
            const switchTo = (offset: number) => {
              const target = siblings[variantIndex - 1 + offset];
              if (!target) return;
              if (!isLastGM) {
                const ok = window.confirm(
                  "このレスポンスより後のやり取りは巻き戻され、元には戻せません。切り替えますか？",
                );
                if (!ok) return;
              }
              onSwitchVariant(target);
            };
            bubble = (
              <GMBubbleRow
                speaker_type={t.speaker_type}
                speaker_name={t.speaker_name}
                is_known={resolveIsKnown(t.speaker_type, t.speaker_name)}
                content={t.content}
                avatarSrc={resolveAvatar(t.speaker_type, t.speaker_name)}
                isGroupTail={isGroupTail}
                isLastGM={isLastGM}
                copyText={groupCopyText}
                onRegenerate={onRegenerate}
                onDiscard={onDiscard}
                onEditCommit={
                  canMutate
                    ? (newContent) => onEditResponse(t.id, newContent)
                    : undefined
                }
                variantIndex={variantIndex}
                variantCount={canMutate ? variantCount : 1}
                onPrevVariant={() => switchTo(-1)}
                onNextVariant={() => switchTo(1)}
                onAvatarClick={
                  npcForAvatar ? () => setNpcDialogTarget(npcForAvatar) : undefined
                }
                elapsedMs={
                  t.id === lastGMTurnId ? elapsedMap?.[t.id] : undefined
                }
                logMessageId={t.log_request_id ?? undefined}
                reasoning={t.reasoning ?? undefined}
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
        {pendingBubbles.map((b, i) => {
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
              isGroupTail={false}
              isLastGM={false}
              onAvatarClick={
                npcForAvatar ? () => setNpcDialogTarget(npcForAvatar) : undefined
              }
              // 生成中のスケッチは先頭バブルの上に 1 つだけ。1 レスポンスが複数の
              // 話者ブロックに割れても、スケッチはそのレスポンス全体に 1 つだから。
              reasoning={i === 0 ? scenarioStreamingReasoning ?? undefined : undefined}
              reasoningStreaming={i === 0}
            />
          );
        })}
        {/* 本文が始まる前のスケッチ（思考が先に流れてくるモデル）。バブルはまだ無いので単独で出す。 */}
        {pendingBubbles.length === 0 && scenarioStreamingReasoning && (
          <div className="max-w-full sm:max-w-[88%]">
            <ThinkingBlock content={scenarioStreamingReasoning} streaming />
          </div>
        )}
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

      {/* ensemble_pc（TRPGモード）専用「ターンを譲る」チップ群。
          ユーザは無言のまま、指定先（他PC枠 / @ALL / @GM）に発話順を回す。
          他PC枠はユーザPC枠を除いた pc_slots（player_type==="character" のもの）。 */}
      {isPcMode && onYieldTo && session.status === "active" && (
        <div
          className="flex flex-col gap-1.5 px-3 py-2 shrink-0"
          style={{ borderTop: "1px solid var(--ch-sep)" }}
        >
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-[10px] text-ch-t3 font-mono shrink-0">
              ターンを譲る:
            </span>
            {pcSlotNames.map((name) => (
              <button
                key={name}
                onClick={() => onYieldTo(name)}
                disabled={sending || editingLocked}
                title={`${name} に発話を譲る`}
                className="flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] transition-colors disabled:opacity-30"
                style={{
                  border: "1px solid var(--ch-sep2)",
                  color: "rgb(var(--ch-t2))",
                }}
              >
                <CharacterAvatar characterName={name} size={16} />
                {name}
              </button>
            ))}
            {pcSlotNames.length > 1 && (
              <button
                onClick={() => onYieldTo("ALL")}
                disabled={sending || editingLocked}
                title="@ALL（PCの誰かに発話を譲る）"
                className="rounded-md px-1.5 py-0.5 text-[11px] transition-colors disabled:opacity-30"
                style={{
                  border: "1px solid var(--ch-sep2)",
                  color: "rgb(var(--ch-t2))",
                }}
              >
                @ALL
              </button>
            )}
            <button
              onClick={() => onYieldTo("GM")}
              disabled={sending || editingLocked}
              title="@GM（語り手に場を進めてもらう）"
              className="rounded-md px-1.5 py-0.5 text-[11px] transition-colors disabled:opacity-30"
              style={{
                border: "1px solid var(--ch-sep2)",
                color: "rgb(var(--ch-t2))",
              }}
            >
              @GM
            </button>
          </div>
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
          placeholder={inputPlaceholder}
          extraTools={mentionToggleButton}
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

/**
 * 編集ロックのスコープを張る薄いラッパ。
 *
 * 本体（Inner）自身も `useEditingLocked()` を読む（宛先トグル・「ターンを譲る」チップの
 * 無効化）ため、Provider は本体の JSX 内ではなく外側に置く必要がある。
 */
export default function ScenarioChatView(props: Props) {
  return (
    <EditingLockProvider>
      <ScenarioChatViewInner {...props} />
    </EditingLockProvider>
  );
}
