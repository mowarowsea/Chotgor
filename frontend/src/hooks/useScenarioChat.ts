/**
 * シナリオプレイ（GM ＋ NPC ＋ あらすじ蒸留）の状態管理・ストリーミング送受信を担うフック。
 *
 * App コンポーネントに散在していたシナリオ系 state（セッション一覧・アクティブセッション・
 * テンプレ・NPC・ターン・未確定吹き出し・あらすじ・進捗・各モーダル開閉）と、その操作
 * ハンドラ（起動・送信・再生成・破棄・編集・あらすじ作成/編集）を 1 箇所へ集約する。
 *
 * 1on1・グループと共有する state（activeSessionId / sending / elapsedMap 等）はフックに
 * 閉じ込めず、引数で setter を受け取る形にして所有権を App 側へ残す。
 */
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";
import {
  deleteScenarioSession,
  deleteScenarioTurnsFrom,
  fetchScenarioSession,
  fetchScenarioSessions,
  fetchScenarioSynopsis,
  fetchScenarioTurns,
  patchScenarioSynopsis,
  regenerateScenarioSynopsis,
  startScenarioSession,
  streamScenarioMessage,
  updateScenarioSession,
} from "../api";
import type {
  PcAssignment,
  ScenarioNpc,
  ScenarioPreset,
  ScenarioSession,
  ScenarioSynopsis,
  ScenarioTemplate,
  ScenarioTurn,
  SynopsisProgress,
} from "../api";
import type { PendingBubble } from "../components/ScenarioChatView";

/** あらすじ作成バーの表示内容（テキストと danger 色フラグ）。 */
interface SynopsisBar {
  text: string;
  danger: boolean;
}

/** useScenarioChat が App から受け取る依存（共有 state の setter とセッション情報）。 */
interface UseScenarioChatDeps {
  /** 現在アクティブなセッション ID。 */
  activeSessionId: string | null;
  /** アクティブセッション ID の setter（起動・選択時に切り替える）。 */
  setActiveSessionId: Dispatch<SetStateAction<string | null>>;
  /** セッション切り替え競合防止用 ref（完了時点での active ID と比較する）。 */
  activeSessionIdRef: MutableRefObject<string | null>;
  /** 送信中フラグの setter。 */
  setSending: Dispatch<SetStateAction<boolean>>;
  /** エラー表示の setter。 */
  setError: (e: string | null) => void;
  /** 経過時間マップの setter。 */
  setElapsedMap: Dispatch<SetStateAction<Record<string, number>>>;
}

/** useScenarioChat が返す state・setter・ハンドラ群。 */
interface UseScenarioChatResult {
  /** シナリオプレイセッション一覧（サイドバーで session に混ぜる）。 */
  scenarioSessions: ScenarioSession[];
  /** scenarioSessions の最新値を参照するための ref（初回ハッシュ復元の同期判定用）。 */
  scenarioSessionsRef: MutableRefObject<ScenarioSession[]>;
  /** 現在選択中のシナリオプレイセッション。 */
  activeScenarioSession: ScenarioSession | null;
  /** 元シナリオテンプレ（場所表示・ユーザPC名(pc_slots[0])表示に使う）。 */
  activeScenarioTemplate: ScenarioTemplate | null;
  /** GM プリセット一覧（gm_preset_id → 表示名解決に使う）。 */
  scenarioPresets: ScenarioPreset[];
  /** アクティブセッションの gm_preset_id を解決した表示用プリセット名。 */
  scenarioPresetName: string | null;
  /** シナリオ NPC 一覧。 */
  scenarioNpcs: ScenarioNpc[];
  /** 確定ターン履歴。 */
  scenarioTurns: ScenarioTurn[];
  /** ストリーミング中の未確定吹き出し列。 */
  scenarioPending: PendingBubble[];
  /** セッションのあらすじ（記憶捏造対策）。未取得は null。 */
  scenarioSynopsis: ScenarioSynopsis | null;
  /** 裏であらすじ蒸留が走っている最中か。 */
  synopsisGenerating: boolean;
  /** あらすじ作成モーダルを表示中か。 */
  synopsisModalOpen: boolean;
  /** シナリオ設定モーダルの開閉状態（null=閉, "model"|"synopsis"=そのタブで開く）。 */
  scenarioSettingsTab: "model" | "synopsis" | null;
  /** あらすじ作成バーの表示内容（非表示は null）。 */
  synopsisBar: SynopsisBar | null;
  /** セッション一覧の setter（初期ロードで使う）。 */
  setScenarioSessions: Dispatch<SetStateAction<ScenarioSession[]>>;
  /** GM プリセット一覧の setter（初期ロードで使う）。 */
  setScenarioPresets: Dispatch<SetStateAction<ScenarioPreset[]>>;
  /** シナリオ設定モーダルの setter（ヘッダーのモデルチップから開く）。 */
  setScenarioSettingsTab: Dispatch<SetStateAction<"model" | "synopsis" | null>>;
  /** シナリオ系 state を初期化する（セッション切り替え時に呼ぶ）。 */
  resetScenarioState: () => void;
  /** シナリオ詳細・ターン・あらすじをロードする（セッション選択時に呼ぶ）。 */
  loadScenarioSession: (sessionId: string) => Promise<void>;
  /** シナリオセッションを削除し一覧から除く。 */
  deleteScenario: (sessionId: string) => Promise<void>;
  /** シナリオテンプレートからプレイセッションを起動する。 */
  handleStartScenario: (
    scenarioId: string,
    gmPresetId: string,
    synopsisPresetId: string,
    title: string | undefined,
    engineType: "ensemble" | "ensemble_pc",
    pcAssignments?: PcAssignment[],
  ) => Promise<void>;
  /** GM プリセットを変更する。 */
  handleScenarioPresetChange: (presetId: string) => Promise<void>;
  /** シナリオ発話送信（SSE ストリーム消費）。
   *  yieldTo は ensemble_pc の「ターンを譲る」UI 用（PC枠名 / "GM" / "ALL"）。
   *  autoAdvance=true と組み合わせて初動ルーティングを直接指定する。 */
  handleScenarioSend: (
    content: string,
    autoAdvance?: boolean,
    regenerateRequestId?: string,
    yieldTo?: string,
  ) => Promise<void>;
  /** ensemble_pc 専用「ターンを譲る」操作。指定先（PC枠名/"GM"/"ALL"）に発話を回す。
   *  内部は handleScenarioSend("", true, undefined, target) のラッパー。 */
  handleScenarioYieldTo: (target: string) => Promise<void>;
  /** GM 応答を 1 レスポンス（= 同一 raw_response 内の話者ブロック群）丸ごと再生成する。 */
  handleScenarioRegenerate: () => Promise<void>;
  /** GM 応答を 1 レスポンス分破棄してユーザ入力待ちに戻す。 */
  handleScenarioDiscard: () => Promise<void>;
  /** ユーザバブルの編集確定（以降を削除して再ストリーム）。 */
  handleScenarioEditUserTurn: (turnId: string, newContent: string) => Promise<void>;
  /** あらすじの部分更新（auto/manual）。 */
  handleSynopsisChange: (patch: { auto?: string; manual?: string }) => Promise<void>;
  /** あらすじ作成（強制蒸留）を裏で起動する。 */
  handleSynopsisCreate: (presetId: string) => void;
  /** あらすじ作成モーダルを開く。 */
  handleOpenSynopsisCreate: () => void;
  /** あらすじ作成モーダルをキャンセルする。 */
  handleCancelSynopsisCreate: () => void;
}

/**
 * シナリオプレイ用フック。
 * @param deps - App が所有する共有 state の setter とセッション情報。
 * @returns シナリオ専用 state・setter・ハンドラ群。
 */
export function useScenarioChat(deps: UseScenarioChatDeps): UseScenarioChatResult {
  const {
    activeSessionId,
    setActiveSessionId,
    activeSessionIdRef,
    setSending,
    setError,
    setElapsedMap,
  } = deps;

  /**
   * シナリオセッション設定モーダル（モデル切替 + あらすじ閲覧/編集を統合）の開閉状態。
   * null で閉、"model" | "synopsis" でそのタブを初期表示して開く。
   */
  const [scenarioSettingsTab, setScenarioSettingsTab] = useState<
    "model" | "synopsis" | null
  >(null);
  /**
   * あらすじ進捗（前回蒸留以降のターン（=話者ブロック）数・文字数と上限）。
   * turn_complete とあらすじ作成（regenerate）時にバックエンドから受け取って更新する。
   * 未取得（レスポンス未完了 / 非シナリオ）は null。バーの表示可否・色はこの比率から導出する。
   */
  const [synopsisProgress, setSynopsisProgress] =
    useState<SynopsisProgress | null>(null);
  /** あらすじ作成モーダルを表示中か（閾値到達時の自動表示 / バナー・設定からの手動表示）。 */
  const [synopsisModalOpen, setSynopsisModalOpen] = useState(false);
  /**
   * 現在の「閾値超え区間」でモーダルを一度閉じた（キャンセルした）か。
   * true の間は後続レスポンスで再び閾値を超えてもモーダルを自動表示しない（うざい再ポップ防止）。
   * 比率が 50% 以下に戻る or あらすじ作成が走ると false へリセットする。
   */
  const [synopsisDismissed, setSynopsisDismissed] = useState(false);
  /** 裏であらすじ蒸留が走っている最中か（控えめなインジケータ表示に使う）。 */
  const [synopsisGenerating, setSynopsisGenerating] = useState(false);

  /** シナリオプレイセッション一覧（サイドバーで session に混ぜる）。 */
  const [scenarioSessions, setScenarioSessions] = useState<ScenarioSession[]>([]);
  /**
   * scenarioSessions の最新値を参照するための ref。
   * 初回マウント時のハッシュ復元など、setState 直後に同期判定が必要な箇所で
   * stale な closure を避けるために使う。
   */
  const scenarioSessionsRef = useRef<ScenarioSession[]>([]);
  scenarioSessionsRef.current = scenarioSessions;
  /** 現在選択中のシナリオプレイセッション。 */
  const [activeScenarioSession, setActiveScenarioSession] = useState<ScenarioSession | null>(null);
  /** 元シナリオテンプレ（場所表示・ユーザPC名(pc_slots[0])表示に使う）。 */
  const [activeScenarioTemplate, setActiveScenarioTemplate] = useState<ScenarioTemplate | null>(null);
  /** GM プリセット一覧（シナリオヘッダーの gm_preset_id → 表示名解決に使う）。 */
  const [scenarioPresets, setScenarioPresets] = useState<ScenarioPreset[]>([]);
  /** アクティブセッションの gm_preset_id を表示用プリセット名に解決する。
   *
   * GM モデルはセッション単位の設定なので、テンプレートではなく ScenarioSession から引く。
   * 同一シナリオから複数セッションを起動した際にそれぞれ別の GM モデルで遊べる。
   */
  const scenarioPresetName = useMemo(() => {
    const id = activeScenarioSession?.gm_preset_id;
    if (!id) return null;
    return scenarioPresets.find((p) => p.id === id)?.name ?? null;
  }, [activeScenarioSession, scenarioPresets]);
  const [scenarioNpcs, setScenarioNpcs] = useState<ScenarioNpc[]>([]);
  const [scenarioTurns, setScenarioTurns] = useState<ScenarioTurn[]>([]);
  /** ストリーミング中の未確定吹き出し列。 */
  const [scenarioPending, setScenarioPending] = useState<PendingBubble[]>([]);
  /** セッションのあらすじ（記憶捏造対策）。未取得は null。 */
  const [scenarioSynopsis, setScenarioSynopsis] = useState<ScenarioSynopsis | null>(null);

  /** シナリオ系 state を初期化する（セッション切り替え・削除時に呼ぶ）。 */
  const resetScenarioState = useCallback(() => {
    setActiveScenarioSession(null);
    setActiveScenarioTemplate(null);
    setScenarioNpcs([]);
    setScenarioTurns([]);
    setScenarioPending([]);
    setScenarioSynopsis(null);
    setSynopsisProgress(null);
    setSynopsisModalOpen(false);
    setSynopsisDismissed(false);
    setSynopsisGenerating(false);
    setScenarioSettingsTab(null);
  }, []);

  /** シナリオ詳細・ターン・あらすじをロードする（セッション選択時に呼ぶ）。 */
  const loadScenarioSession = useCallback(async (sessionId: string) => {
    try {
      const [detail, ts, syn] = await Promise.all([
        fetchScenarioSession(sessionId),
        fetchScenarioTurns(sessionId),
        fetchScenarioSynopsis(sessionId).catch(() => null),
      ]);
      setActiveScenarioSession(detail);
      setActiveScenarioTemplate(detail.scenario);
      setScenarioNpcs(detail.npcs);
      setScenarioTurns(ts);
      setScenarioSynopsis(syn);
    } catch (e) {
      setError(String(e));
    }
  }, [setError]);

  /** シナリオセッションを削除し一覧から除く。 */
  const deleteScenario = useCallback(async (sessionId: string) => {
    await deleteScenarioSession(sessionId);
    setScenarioSessions((prev) => prev.filter((s) => s.id !== sessionId));
  }, []);

  /** シナリオテンプレートからプレイセッションを起動する。
   *
   * `gmPresetId` は GM を演じる LLM プリセット（必須）、`synopsisPresetId` は
   * あらすじ蒸留専用の LLM プリセット（必須・同じプリセットでもよい）。
   * NewSessionPicker の Scenario タブでユーザに両方選ばせる。
   */
  const handleStartScenario = useCallback(
    async (
      scenarioId: string,
      gmPresetId: string,
      synopsisPresetId: string,
      title: string | undefined,
      engineType: "ensemble" | "ensemble_pc",
      pcAssignments?: PcAssignment[],
    ) => {
      setError(null);
      try {
        const created = await startScenarioSession(
          scenarioId,
          gmPresetId,
          synopsisPresetId,
          title,
          engineType,
          pcAssignments,
        );
        setScenarioSessions((prev) => [created, ...prev]);
        // ref も即座に更新する。直後の handleSelectSession 等が ref を読むケースに備える。
        scenarioSessionsRef.current = [created, ...scenarioSessionsRef.current];
        // 起動と同時に詳細＋ターン履歴を取り直す。
        // バックエンドは scenario.intro を起動時に固定ターンとして挿入するため、
        // ここで turns を fetch しないと intro が画面に出ない（リロード後にだけ見える）。
        const [detail, initialTurns, initialSyn] = await Promise.all([
          fetchScenarioSession(created.id),
          fetchScenarioTurns(created.id),
          fetchScenarioSynopsis(created.id).catch(() => null),
        ]);
        setActiveSessionId(created.id);
        setActiveScenarioSession(detail);
        setActiveScenarioTemplate(detail.scenario);
        setScenarioNpcs(detail.npcs);
        setScenarioTurns(initialTurns);
        setScenarioPending([]);
        setScenarioSynopsis(initialSyn);
        window.location.hash = created.id;
      } catch (e) {
        setError(String(e));
      }
    },
    [setActiveSessionId, setError],
  );

  /** シナリオセッションの GM プリセットを変更する。
   *
   * 左上ヘッダーのモーダル「シナリオ用モデル」タブから呼ばれる。次レスポンス以降の GM 応答に
   * 新プリセットが反映される。あらすじ蒸留モデルとは独立。
   */
  const handleScenarioPresetChange = useCallback(
    async (presetId: string) => {
      if (!activeScenarioSession) return;
      if (presetId === activeScenarioSession.gm_preset_id) return;
      setError(null);
      try {
        const updated = await updateScenarioSession(activeScenarioSession.id, {
          gm_preset_id: presetId,
        });
        // セッション一覧側にも反映する（ヘッダー用 state は session + 一覧の両方を更新）。
        setActiveScenarioSession((prev) =>
          prev ? { ...prev, gm_preset_id: updated.gm_preset_id } : prev,
        );
        setScenarioSessions((prev) =>
          prev.map((s) =>
            s.id === updated.id ? { ...s, gm_preset_id: updated.gm_preset_id } : s,
          ),
        );
      } catch (e) {
        setError(String(e));
      }
    },
    [activeScenarioSession, setError],
  );

  /**
   * シナリオ発話送信。SSE ストリームを消費しながら吹き出しを更新する。
   *
   * autoAdvance=true なら「ユーザは無言で続きを促す」モード。
   * content は何が来てもサーバ側で無視され、user turn も保存されない。
   * regenerateRequestId を指定すると、再生成ログを同一エントリにまとめる。
   */
  const handleScenarioSend = useCallback(
    async (
      content: string,
      autoAdvance: boolean = false,
      regenerateRequestId?: string,
      yieldTo?: string,
    ) => {
      if (!activeScenarioSession) return;
      setError(null);
      setSending(true);
      setScenarioPending([]);
      // モデルへリクエスト〜turn 完了までの経過時間を計測する開始時刻。
      const turnStartedAt = performance.now();
      try {
        const sessionId = activeScenarioSession.id;
        const newPending: PendingBubble[] = [];
        for await (const ev of streamScenarioMessage(
          sessionId,
          content,
          autoAdvance,
          regenerateRequestId,
          yieldTo,
        )) {
          if (ev.type === "user_saved") {
            // user_saved はユーザ発話を確定ターンとしてリストに追加する
            setScenarioTurns((prev) => [...prev, ev.turn]);
          } else if (ev.type === "speaker_start") {
            // 新しい吹き出しを未確定として追加する。
            // 安定キー (`id`)をここで一度だけ発行することで、後段の shift で
            // インデックスがズレても他の pending バブルが React 上で再マウントされない。
            newPending.push({
              id:
                typeof crypto !== "undefined" && "randomUUID" in crypto
                  ? crypto.randomUUID()
                  : `pending-${Date.now()}-${Math.random().toString(36).slice(2)}`,
              speaker_type: ev.speaker_type,
              speaker_name: ev.speaker_name,
              speaker_id: ev.speaker_id,
              is_known: ev.is_known,
              content: "",
            });
            setScenarioPending([...newPending]);
          } else if (ev.type === "content_delta") {
            // 最新の未確定吹き出しに本文を追記する
            if (newPending.length > 0) {
              newPending[newPending.length - 1].content += ev.text;
              setScenarioPending([...newPending]);
            }
          } else if (ev.type === "speaker_end") {
            // 確定ターンとしてリストに追加し、対応する未確定吹き出しを除く
            setScenarioTurns((prev) => [...prev, ev.turn]);
            if (newPending.length > 0) newPending.shift();
            setScenarioPending([...newPending]);
          } else if (ev.type === "pc_start") {
            // ensemble_pc 専用: GM レスポンス後に PC（Chotgorキャラ）が応答開始する。
            // 既存 speaker_start と同じ pendingBubble 機構に乗せる（PC は character として描画）。
            newPending.push({
              id:
                typeof crypto !== "undefined" && "randomUUID" in crypto
                  ? crypto.randomUUID()
                  : `pending-${Date.now()}-${Math.random().toString(36).slice(2)}`,
              speaker_type: "pc",
              speaker_name: ev.character,
              speaker_id: ev.character_id,
              is_known: true,
              content: "",
            });
            setScenarioPending([...newPending]);
          } else if (ev.type === "pc_chunk") {
            // PC 発話テキストを最新 pending bubble に追記する（speaker_end までに完成する）
            if (newPending.length > 0) {
              newPending[newPending.length - 1].content += ev.content;
              setScenarioPending([...newPending]);
            }
          } else if (ev.type === "pc_reasoning") {
            // 想起記憶・WM スレッド・思考ブロックは現状 UI に出さない（将来 reasoning パネルへ）
          } else if (ev.type === "pc_done") {
            // PC レスポンス完了の最終通知。本文の確定は後続の speaker_end が行うため、ここは何もしない
          } else if (ev.type === "pc_error") {
            setError(`${ev.character}: ${ev.message}`);
          } else if (ev.type === "pc_angle_switched") {
            // PC が switch_angle した。セッションへの永続化は backend 側未実装のためログのみ
            console.info(
              "[scenario_pc] angle switched",
              ev.character,
              "→",
              ev.preset_name,
            );
          } else if (ev.type === "turn_complete") {
            // ユーザターン完了（GM/PC のレスポンス連鎖が終わり、ユーザ入力待ちへ戻った）。
            // 残った未確定吹き出しは捨てる（speaker_end でほぼ消えるはず）。
            newPending.length = 0;
            setScenarioPending([]);
            // 経過時間を記録する。同一レスポンス内の全GMバブル（複数話者ブロック）に同じ値を共有する。
            const elapsed = performance.now() - turnStartedAt;
            if (ev.turn_ids.length > 0) {
              setElapsedMap((prev) => {
                const next = { ...prev };
                for (const tid of ev.turn_ids) next[tid] = elapsed;
                return next;
              });
            }
          } else if (ev.type === "synopsis_progress") {
            // ユーザターン完了直後の進捗。バーの表示/色とモーダル自動表示は
            // synopsisProgress を監視する useEffect 側で判定する。
            setSynopsisProgress({
              turns: ev.turns,
              max_turns: ev.max_turns,
              chars: ev.chars,
              max_chars: ev.max_chars,
            });
          } else if (ev.type === "error") {
            setError(ev.message);
            break;
          } else if (ev.type === "done") {
            break;
          }
        }
        // 完了後にサーバから真の turns を取り直して整合性確保
        if (sessionId === activeSessionIdRef.current) {
          try {
            const ts = await fetchScenarioTurns(sessionId);
            setScenarioTurns(ts);
          } catch {
            // 取得失敗は無視
          }
          // テンプレートとセッション本体を取り直す（updated_at や設定変更を反映）。
          // GM プリセット変更はセッション側の更新で即時反映されるが、
          // テンプレ側を別タブで編集した場合に追従するためここでも取り直す。
          fetchScenarioSession(sessionId)
            .then((d) => {
              if (sessionId === activeSessionIdRef.current) {
                setActiveScenarioSession(d);
                setActiveScenarioTemplate(d.scenario);
              }
            })
            .catch(() => {});
          // セッション一覧も最新化（updated_at 反映）
          fetchScenarioSessions().then(setScenarioSessions).catch(() => {});
        }
      } catch (e) {
        setError(String(e));
      } finally {
        setScenarioPending([]);
        setSending(false);
      }
    },
    [activeScenarioSession, activeSessionIdRef, setSending, setError, setElapsedMap],
  );

  /**
   * シナリオの GM 応答を 1 レスポンス（=同一 raw_response 内の話者ブロック群）丸ごと再生成する。
   *
   * レスポンス境界は `raw_response` を共有する連続バブル列で判定する:
   *   - GM の 1 回の LLM 呼出 = 同一 raw_response の GM バブル列（複数ターン=話者ブロックを含みうる）
   *   - その直前に user 発話があれば通常レスポンス → user 起点で再ストリーム
   *   - 直前に user 発話がなければ auto_advance レスポンス → GM 列の先頭から
   *     auto_advance=true で再ストリーム
   *
   * 1on1 chat の retry と同じ思想で、「レスポンスの開始点」より後を捨てる方式。
   */
  const handleScenarioRegenerate = useCallback(async () => {
    if (!activeScenarioSession) return;
    if (scenarioTurns.length === 0) return;

    // 末尾 GM 列の先頭 index を raw_response 連続性で探す。
    // 末尾が user の場合（GM 応答待ち状態）はそのまま user を起点にする。
    let lastTurnStart = scenarioTurns.length - 1;
    if (scenarioTurns[lastTurnStart].speaker_type !== "user") {
      const tailRaw = scenarioTurns[lastTurnStart].raw_response;
      while (lastTurnStart > 0) {
        const prev = scenarioTurns[lastTurnStart - 1];
        if (prev.speaker_type === "user") break;
        if (prev.raw_response !== tailRaw) break;
        lastTurnStart--;
      }
    }

    // 直前に user 発話があるかを見て、通常 / auto_advance を判別。
    const prev = lastTurnStart > 0 ? scenarioTurns[lastTurnStart - 1] : null;
    // 再生成対象の先頭 GM ターン（=先頭話者ブロック）の log_request_id を引き継ぐ（ログをまとめるため）
    const gmLogRequestId =
      scenarioTurns[lastTurnStart]?.speaker_type !== "user"
        ? (scenarioTurns[lastTurnStart]?.log_request_id ?? undefined)
        : undefined;
    let pivot: ScenarioTurn;
    let resend: () => Promise<void>;
    if (prev && prev.speaker_type === "user") {
      // 通常レスポンス: user を含めて削除し、同じ発話で再ストリーム
      pivot = prev;
      const content = prev.content;
      resend = () => handleScenarioSend(content, false, gmLogRequestId);
    } else if (scenarioTurns[lastTurnStart].speaker_type !== "user") {
      // auto_advance レスポンス: GM 列先頭から削除して auto_advance で再ストリーム
      pivot = scenarioTurns[lastTurnStart];
      resend = () => handleScenarioSend("", true, gmLogRequestId);
    } else {
      // 末尾が user で GM 応答が無い特殊状態（前回ストリームエラー後など）
      pivot = scenarioTurns[lastTurnStart];
      const content = pivot.content;
      resend = () => handleScenarioSend(content, false);
    }

    const pivotIndex = pivot.turn_index;
    try {
      await deleteScenarioTurnsFrom(activeScenarioSession.id, pivot.id);
      setScenarioTurns((prevTurns) =>
        prevTurns.filter((t) => t.turn_index < pivotIndex),
      );
      await resend();
    } catch (e) {
      setError(String(e));
    }
  }, [activeScenarioSession, scenarioTurns, handleScenarioSend, setError]);

  /**
   * シナリオの GM 応答を 1 レスポンス分破棄してユーザ入力待ちに戻す。
   *
   * `handleScenarioRegenerate` と異なり、削除後に再ストリームしない。
   * 主な用途: ユーザが auto_advance（無入力 Enter）で GM 続きを促した結果を
   * 気に入らず、その GM 応答を捨てて自分で発話を入力したい場合。
   *
   * 削除対象は末尾 GM 列のみ（同一 raw_response のバブル列）。
   * 直前のユーザ発話があれば残す（そこから次の発話を入力できる）。
   * 末尾が user の状態（GM 未応答）では何もしない。
   */
  const handleScenarioDiscard = useCallback(async () => {
    if (!activeScenarioSession) return;
    if (scenarioTurns.length === 0) return;

    const lastIndex = scenarioTurns.length - 1;
    if (scenarioTurns[lastIndex].speaker_type === "user") return;

    // 末尾 GM 列の先頭を raw_response の連続性で探す
    let groupStart = lastIndex;
    const tailRaw = scenarioTurns[lastIndex].raw_response;
    while (groupStart > 0) {
      const prev = scenarioTurns[groupStart - 1];
      if (prev.speaker_type === "user") break;
      if (prev.raw_response !== tailRaw) break;
      groupStart--;
    }

    const pivot = scenarioTurns[groupStart];
    const pivotIndex = pivot.turn_index;
    try {
      await deleteScenarioTurnsFrom(activeScenarioSession.id, pivot.id);
      setScenarioTurns((prevTurns) =>
        prevTurns.filter((t) => t.turn_index < pivotIndex),
      );
    } catch (e) {
      setError(String(e));
    }
  }, [activeScenarioSession, scenarioTurns, setError]);

  /**
   * ユーザバブルの編集確定処理。
   *
   * 編集対象 user turn 以降を全削除し、新しい内容で再ストリームする。
   * GM 応答の再生成と同じ流れ。
   */
  const handleScenarioEditUserTurn = useCallback(
    async (turnId: string, newContent: string) => {
      if (!activeScenarioSession) return;
      const target = scenarioTurns.find((t) => t.id === turnId);
      if (!target) return;
      try {
        await deleteScenarioTurnsFrom(activeScenarioSession.id, turnId);
        setScenarioTurns((prev) =>
          prev.filter((t) => t.turn_index < target.turn_index),
        );
        await handleScenarioSend(newContent);
      } catch (e) {
        setError(String(e));
      }
    },
    [activeScenarioSession, scenarioTurns, handleScenarioSend, setError],
  );

  /** あらすじの部分更新（auto/manual）。ScenarioSettingsModal から呼ばれる。 */
  const handleSynopsisChange = useCallback(
    async (patch: { auto?: string; manual?: string }) => {
      if (!activeScenarioSession) return;
      try {
        const updated = await patchScenarioSynopsis(activeScenarioSession.id, patch);
        setScenarioSynopsis(updated);
      } catch (e) {
        setError(String(e));
      }
    },
    [activeScenarioSession, setError],
  );

  /**
   * あらすじ作成（強制蒸留）を裏で起動する。あらすじ作成モーダルの「作成」から呼ばれる。
   *
   * 旧設計はターン開始前に同期蒸留していたが、本フローは非ブロッキング。モーダルを閉じ、
   * 控えめなインジケータを出してから蒸留を走らせ、その間もユーザはチャットを続けられる。
   * 選んだ preset はサーバ側でセッションへ永続化（記憶）されるため、ローカルにも反映する。
   */
  const handleSynopsisCreate = useCallback(
    (presetId: string) => {
      if (!activeScenarioSession) return;
      const sessionId = activeScenarioSession.id;
      setSynopsisModalOpen(false);
      setSynopsisDismissed(false);
      setSynopsisGenerating(true);
      // 選択 preset をローカルの session にも反映（次回モーダルの初期選択に効く）。
      setActiveScenarioSession((prev) =>
        prev && prev.id === sessionId
          ? { ...prev, synopsis_preset_id: presetId }
          : prev,
      );
      setScenarioSessions((prev) =>
        prev.map((s) =>
          s.id === sessionId ? { ...s, synopsis_preset_id: presetId } : s,
        ),
      );
      regenerateScenarioSynopsis(sessionId, presetId)
        .then((res) => {
          if (sessionId === activeSessionIdRef.current) {
            // 蒸留後の synopsis と最新進捗で反映。進捗は通常 0 に戻りバーが消える。
            setScenarioSynopsis(res.synopsis);
            setSynopsisProgress(res.progress);
          }
        })
        .catch((e) => setError(String(e)))
        .finally(() => setSynopsisGenerating(false));
    },
    [activeScenarioSession, activeSessionIdRef, setError],
  );

  /**
   * ensemble_pc の「ターンを譲る」操作。ユーザは無言のまま、指定先へ初動を回す。
   *
   * 内部は `handleScenarioSend("", autoAdvance=true, undefined, target)` のラッパー。
   * target に PC枠名を渡せばその PC、"GM" なら GM、"ALL" ならランダム PC へルーティングされる。
   */
  const handleScenarioYieldTo = useCallback(
    async (target: string) => {
      await handleScenarioSend("", true, undefined, target);
    },
    [handleScenarioSend],
  );

  /** あらすじ作成モーダルを開く（バー / 設定モーダルの「自動作成」から呼ばれる）。 */
  const handleOpenSynopsisCreate = useCallback(() => {
    setScenarioSettingsTab(null);
    setSynopsisModalOpen(true);
  }, []);

  /** あらすじ作成モーダルをキャンセルする。以降はバーで作成を促し続ける（再ポップしない）。 */
  const handleCancelSynopsisCreate = useCallback(() => {
    setSynopsisModalOpen(false);
    setSynopsisDismissed(true);
  }, []);

  /**
   * あらすじ進捗（前回蒸留以降のターン数・文字数）と上限から比率を求める。
   * ターン側・文字側それぞれの達成率のうち高い方（より限界に近い方）を採用する。
   */
  const synopsisRatio = useMemo(() => {
    if (!synopsisProgress) return 0;
    const { turns, max_turns, chars, max_chars } = synopsisProgress;
    const rt = max_turns > 0 ? turns / max_turns : 0;
    const rc = max_chars > 0 ? chars / max_chars : 0;
    return Math.max(rt, rc);
  }, [synopsisProgress]);

  /**
   * あらすじ作成バーの表示内容。比率が 50% 以下、生成中、モーダル表示中は null（非表示）。
   * テキストはターン側・文字側のうち限界に近い方を「あらすじ未作成（X/Y…）」で表示し、
   * 80% を超えたら danger（赤）にする。
   */
  const synopsisBar = useMemo<SynopsisBar | null>(() => {
    if (!synopsisProgress) return null;
    if (synopsisGenerating || synopsisModalOpen) return null;
    if (synopsisRatio <= 0.5) return null;
    const { turns, max_turns, chars, max_chars } = synopsisProgress;
    const rt = max_turns > 0 ? turns / max_turns : 0;
    const rc = max_chars > 0 ? chars / max_chars : 0;
    const text =
      rt >= rc
        ? `あらすじ未作成（${turns}/${max_turns}ターン）`
        : `あらすじ未作成（${chars}/${max_chars}文字）`;
    return { text, danger: synopsisRatio > 0.8 };
  }, [synopsisProgress, synopsisGenerating, synopsisModalOpen, synopsisRatio]);

  /**
   * あらすじ進捗の変化に応じて作成モーダルの自動表示を制御する。
   * - 比率 50% 以下: 「閾値超え区間」が終了したものとして dismissed をリセット
   * - 比率 50% 超 かつ 未 dismissed・非生成中・未表示: 作成モーダルを自動表示
   */
  useEffect(() => {
    if (!synopsisProgress) return;
    if (synopsisRatio <= 0.5) {
      if (synopsisDismissed) setSynopsisDismissed(false);
      return;
    }
    if (!synopsisDismissed && !synopsisGenerating && !synopsisModalOpen) {
      setSynopsisModalOpen(true);
    }
  }, [
    synopsisProgress,
    synopsisRatio,
    synopsisDismissed,
    synopsisGenerating,
    synopsisModalOpen,
  ]);

  return {
    scenarioSessions,
    scenarioSessionsRef,
    activeScenarioSession,
    activeScenarioTemplate,
    scenarioPresets,
    scenarioPresetName,
    scenarioNpcs,
    scenarioTurns,
    scenarioPending,
    scenarioSynopsis,
    synopsisGenerating,
    synopsisModalOpen,
    scenarioSettingsTab,
    synopsisBar,
    setScenarioSessions,
    setScenarioPresets,
    setScenarioSettingsTab,
    resetScenarioState,
    loadScenarioSession,
    deleteScenario,
    handleStartScenario,
    handleScenarioPresetChange,
    handleScenarioSend,
    handleScenarioYieldTo,
    handleScenarioRegenerate,
    handleScenarioDiscard,
    handleScenarioEditUserTurn,
    handleSynopsisChange,
    handleSynopsisCreate,
    handleOpenSynopsisCreate,
    handleCancelSynopsisCreate,
  };
}
