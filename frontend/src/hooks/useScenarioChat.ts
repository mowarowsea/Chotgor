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
  activateScenarioGeneration,
  deleteScenarioSession,
  deleteScenarioTurnsFrom,
  fetchScenarioSession,
  fetchScenarioSessions,
  fetchScenarioSynopsis,
  fetchScenarioTurns,
  patchScenarioSynopsis,
  patchScenarioTurn,
  regenerateScenarioSynopsis,
  startScenarioSession,
  streamScenarioMessage,
  updateScenarioSession,
} from "../api";
import { consumeStream } from "./streamingUtils";
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

/**
 * 履歴ウィンドウ 1 ページぶんのターン数（初回表示・遡り読み込み共通）。
 *
 * 数百ターン規模のセッションで全件を取り直すと、転送・パース・全バブルの再構築が
 * 毎回走って体感が重くなる。表示は直近ウィンドウに限り、過去は明示操作で伸ばす。
 */
const TURN_PAGE_SIZE = 60;

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
  /** PC ターンの log_message_id を turn.id に紐付けるための共有 setter（1on1 と統合）。 */
  setMsgLogIds: Dispatch<SetStateAction<Record<string, string>>>;
}

/**
 * シナリオ発話送信（ストリーム消費）の結果。
 *
 * 引き直しの失敗を呼び出し側が検知して巻き戻しを戻すために返す。
 * ストリームは `error` イベントでも例外を投げずに終わるため、成否は戻り値で伝える。
 */
interface ScenarioSendResult {
  /** ストリームが error イベント・例外なしで終わったか。 */
  ok: boolean;
  /** この送信で保存された最初のターンID（1件も保存されなければ null）。
   *  失敗した試行を後片付けするときの削除起点に使う。 */
  firstSavedTurnId: string | null;
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
  /** 確定ターン履歴（直近ウィンドウぶんのみ。過去は遡り読み込みで伸びる）。 */
  scenarioTurns: ScenarioTurn[];
  /** 表示中ウィンドウより過去のターンがまだ残っているか。 */
  hasOlderTurns: boolean;
  /** 遡り読み込みの実行中フラグ。 */
  loadingOlderTurns: boolean;
  /** 表示ウィンドウを 1 ページぶん過去へ伸ばす。 */
  loadOlderScenarioTurns: () => Promise<void>;
  /** ストリーミング中の未確定吹き出し列。 */
  scenarioPending: PendingBubble[];
  /** 生成中レスポンスの reasoning（想起記憶・WM・スケッチ）。1on1 の streamingReasoning と同じ役割。
   *  確定後は `ScenarioTurn.reasoning`（DB 保存）が表示を引き継ぐため、turn_end で null に戻す。 */
  scenarioStreamingReasoning: string | null;
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
    yieldTo?: string,
  ) => Promise<ScenarioSendResult>;
  /** ensemble_pc 専用「ターンを譲る」操作。指定先（PC枠名/"GM"/"ALL"）に発話を回す。
   *  内部は handleScenarioSend("", true, undefined, target) のラッパー。 */
  handleScenarioYieldTo: (target: string) => Promise<void>;
  /** GM 応答を 1 レスポンス（= 同一 response_key の話者ブロック群）丸ごと再生成する。 */
  handleScenarioRegenerate: () => Promise<void>;
  /** GM 応答を 1 レスポンス分破棄してユーザ入力待ちに戻す。 */
  handleScenarioDiscard: () => Promise<void>;
  /** ユーザバブルの編集確定（以降を枝ごと削除して再ストリーム）。 */
  handleScenarioEditUserTurn: (turnId: string, newContent: string) => Promise<void>;
  /** 末尾ユーザ発話の削除（再ストリームなし）。 */
  handleScenarioDeleteUserTurn: (turnId: string) => Promise<void>;
  /** 枝（レスポンスガチャ）の切替。分岐点より後の本線は巻き戻される。 */
  handleScenarioSwitchVariant: (generationId: string) => Promise<void>;
  /** GM / PC 発話の手動上書き（枝は生やさず本文だけ差し替える）。 */
  handleScenarioEditResponse: (turnId: string, newContent: string) => Promise<void>;
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
    setMsgLogIds,
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
  /**
   * scenarioTurns の最新値を参照するための ref。
   * 送信完了後のマージなど、setState のクロージャ外で現在の件数を見たい箇所で使う。
   */
  const scenarioTurnsRef = useRef<ScenarioTurn[]>([]);
  scenarioTurnsRef.current = scenarioTurns;
  /** 表示中ウィンドウより過去のターンがまだ残っているか（遡りボタンの表示可否）。 */
  const [hasOlderTurns, setHasOlderTurns] = useState(false);
  /** 遡り読み込みの実行中フラグ（多重発火の抑止と表示用）。 */
  const [loadingOlderTurns, setLoadingOlderTurns] = useState(false);
  /**
   * 送信の世代カウンタ。`turn_complete` で `sending` を早めに落とすため、
   * 完了後の整合性再取得が着弾する前にユーザが次を送れる。古い送信の後処理が
   * 新しい送信の結果を上書きしないよう、着弾時にこの値の一致を確認する。
   */
  const sendSeqRef = useRef(0);
  /** ストリーミング中の未確定吹き出し列。 */
  const [scenarioPending, setScenarioPending] = useState<PendingBubble[]>([]);
  /** 生成中レスポンスの reasoning（想起記憶・WM スレッド・スケッチ）。確定ターンぶんは
   *  `ScenarioTurn.reasoning` が持つので、ここは「まだ確定していない今のぶん」だけを持つ。 */
  const [scenarioStreamingReasoning, setScenarioStreamingReasoning] = useState<string | null>(null);
  /** セッションのあらすじ（記憶捏造対策）。未取得は null。 */
  const [scenarioSynopsis, setScenarioSynopsis] = useState<ScenarioSynopsis | null>(null);

  /**
   * 取得した直近ウィンドウを表示に反映する。
   *
   * 「まだ上があるか」は取得件数がページ幅ちょうどかで近似する。実際には
   * ぴったり尽きていた場合に空振りの追加取得が 1 回起きるが、実害はない。
   */
  const applyTurnWindow = useCallback((ts: ScenarioTurn[]) => {
    setScenarioTurns(ts);
    setHasOlderTurns(ts.length === TURN_PAGE_SIZE);
  }, []);

  /**
   * 直近ウィンドウの取得結果を、遡り読み込み済みの過去を保ったまま反映する。
   *
   * レスポンス完了後の整合性再取得で使う。`applyTurnWindow` のように丸ごと
   * 差し替えると、ユーザが遡って読み込んだ履歴が送信のたびに消えてしまう。
   * 再取得で更新したいのは末尾側だけなので、取得ウィンドウの先頭 `turn_index` を
   * 境に、それより古い表示中のターンはそのまま残す。
   *
   * 枝の切替では分岐点より後が巻き戻り、前方の並びも変わりうるため、こちらではなく
   * `applyTurnWindow` でウィンドウごとリセットする。
   */
  const mergeTurnWindow = useCallback((fresh: ScenarioTurn[]) => {
    if (fresh.length === 0) return;
    const cut = fresh[0].turn_index;
    // 初回（表示が空）だけは上端判定を更新する。既に何か表示していれば、
    // 前方を保持している以上「まだ上があるか」の答えは変わらない。
    if (scenarioTurnsRef.current.length === 0) {
      setHasOlderTurns(fresh.length === TURN_PAGE_SIZE);
    }
    setScenarioTurns((prev) => [
      ...prev.filter((t) => t.turn_index < cut),
      ...fresh,
    ]);
  }, []);

  /** シナリオ系 state を初期化する（セッション切り替え・削除時に呼ぶ）。 */
  const resetScenarioState = useCallback(() => {
    setActiveScenarioSession(null);
    setActiveScenarioTemplate(null);
    setScenarioNpcs([]);
    setScenarioTurns([]);
    setHasOlderTurns(false);
    setLoadingOlderTurns(false);
    setScenarioPending([]);
    setScenarioStreamingReasoning(null);
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
        fetchScenarioTurns(sessionId, { limit: TURN_PAGE_SIZE }),
        fetchScenarioSynopsis(sessionId).catch(() => null),
      ]);
      setActiveScenarioSession(detail);
      setActiveScenarioTemplate(detail.scenario);
      setScenarioNpcs(detail.npcs);
      applyTurnWindow(ts);
      setScenarioSynopsis(syn);
    } catch (e) {
      setError(String(e));
    }
  }, [applyTurnWindow, setError]);

  /**
   * 表示ウィンドウを 1 ページぶん過去へ伸ばす（「以前のやり取りを読み込む」）。
   *
   * 現在の先頭ターンより手前を取得して前方に連結する。取得済みと重ならないよう
   * `before_index` を境界に使うので、多重呼び出しでも重複はしない。
   */
  const loadOlderScenarioTurns = useCallback(async () => {
    if (!activeScenarioSession) return;
    if (loadingOlderTurns || !hasOlderTurns) return;
    const oldest = scenarioTurns[0];
    if (!oldest) return;
    setLoadingOlderTurns(true);
    try {
      const older = await fetchScenarioTurns(activeScenarioSession.id, {
        limit: TURN_PAGE_SIZE,
        beforeIndex: oldest.turn_index,
      });
      setScenarioTurns((prev) => [...older, ...prev]);
      setHasOlderTurns(older.length === TURN_PAGE_SIZE);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoadingOlderTurns(false);
    }
  }, [
    activeScenarioSession,
    scenarioTurns,
    hasOlderTurns,
    loadingOlderTurns,
    setError,
  ]);

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
          fetchScenarioTurns(created.id, { limit: TURN_PAGE_SIZE }),
          fetchScenarioSynopsis(created.id).catch(() => null),
        ]);
        setActiveSessionId(created.id);
        setActiveScenarioSession(detail);
        setActiveScenarioTemplate(detail.scenario);
        setScenarioNpcs(detail.npcs);
        applyTurnWindow(initialTurns);
        setScenarioPending([]);
        setScenarioSynopsis(initialSyn);
        window.location.hash = created.id;
      } catch (e) {
        setError(String(e));
      }
    },
    [applyTurnWindow, setActiveSessionId, setError],
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
   */
  const handleScenarioSend = useCallback(
    async (
      content: string,
      autoAdvance: boolean = false,
      yieldTo?: string,
    ) => {
      if (!activeScenarioSession) return { ok: false, firstSavedTurnId: null };
      setError(null);
      setSending(true);
      setScenarioPending([]);
      setScenarioStreamingReasoning(null);
      // ストリームの成否と、この送信で最初に保存されたターン。引き直しが失敗したとき、
      // 途中まで保存された分を片付けて巻き戻しを戻すために呼び出し側が使う。
      let ok = true;
      let firstSavedTurnId: string | null = null;
      // この送信の世代。完了後の後処理が着弾する頃に次の送信が始まっていたら、
      // 古い結果で上書きしないための番号（`sendSeqRef` の説明を参照）。
      const seq = ++sendSeqRef.current;
      const isLatestSend = () => seq === sendSeqRef.current;
      // モデルへリクエスト〜turn 完了までの経過時間を計測する開始時刻。
      const turnStartedAt = performance.now();
      // 生成中レスポンスの reasoning。GM・PC いずれも `reasoning` イベントでここへ溜まり、
      // 未確定バブルの上にライブ表示される。確定（turn_end）時点で DB 保存済みの
      // `ScenarioTurn.reasoning` へ表示が引き継がれるので、ここは空へ戻す。
      let pendingReasoning = "";
      // log_message_id は pc_done で届き、直後の turn_end で turn.id へ紐付ける。
      let pendingPcLogMessageId: string | null = null;
      const sessionId = activeScenarioSession.id;
      const newPending: PendingBubble[] = [];

      await consumeStream({
        stream: streamScenarioMessage(
          sessionId,
          content,
          autoAdvance,
          yieldTo,
        ),
        onEvent: (ev) => {
          if (ev.type === "user_saved") {
            // user_saved はユーザ発話を確定ターンとしてリストに追加する
            if (!firstSavedTurnId) firstSavedTurnId = ev.turn.id;
            setScenarioTurns((prev) => [...prev, ev.turn]);
          } else if (ev.type === "turn_start") {
            // 新しい吹き出しを未確定として追加する。GM (speaker_type 有) と PC (character 有)
            // で payload 差分があるが、いずれも同じ pendingBubble 機構に乗せる。
            // 安定キー (`id`)をここで一度だけ発行することで、後段の shift で
            // インデックスがズレても他の pending バブルが React 上で再マウントされない。
            const isPcTurn = "character" in ev;
            newPending.push({
              id:
                typeof crypto !== "undefined" && "randomUUID" in crypto
                  ? crypto.randomUUID()
                  : `pending-${Date.now()}-${Math.random().toString(36).slice(2)}`,
              speaker_type: isPcTurn ? "pc" : ev.speaker_type,
              speaker_name: isPcTurn ? ev.character : ev.speaker_name,
              speaker_id: isPcTurn ? ev.character_id : ev.speaker_id,
              is_known: isPcTurn ? true : ev.is_known,
              content: "",
            });
            setScenarioPending([...newPending]);
          } else if (ev.type === "chunk") {
            // 最新の未確定吹き出しに本文を追記する（GM は text、PC は content フィールド）。
            if (newPending.length > 0) {
              const text = "text" in ev ? ev.text : ev.content;
              newPending[newPending.length - 1].content += text;
              setScenarioPending([...newPending]);
            }
          } else if (ev.type === "reasoning") {
            // 想起記憶・WM スレッド・スケッチ。GM（character なし）と PC（character あり）で
            // 同じ扱い ── いま生成中のレスポンス 1 つぶんとして溜め、ライブ表示する。
            pendingReasoning += ev.content;
            setScenarioStreamingReasoning(pendingReasoning);
          } else if (ev.type === "turn_end") {
            // 確定ターンとしてリストに追加し、対応する未確定吹き出しを除く
            if (!firstSavedTurnId) firstSavedTurnId = ev.turn.id;
            setScenarioTurns((prev) => [...prev, ev.turn]);
            if (newPending.length > 0) newPending.shift();
            setScenarioPending([...newPending]);
            // 確定ターンが reasoning を持って返ってくる（GM はレスポンス先頭ターン、PC は自分自身）。
            // ライブ表示の役目はここで終わりなので畳む。
            pendingReasoning = "";
            setScenarioStreamingReasoning(null);
            // 直前 PC ターンの log_message_id が溜まっていれば、この turn_end の turn.id へ紐付ける。
            if (ev.turn.speaker_type === "pc" && pendingPcLogMessageId) {
              const turnId = ev.turn.id;
              const logId = pendingPcLogMessageId;
              setMsgLogIds((prev) => ({ ...prev, [turnId]: logId }));
              pendingPcLogMessageId = null;
            }
          } else if (ev.type === "pc_done") {
            // PC レスポンス完了通知。log_message_id を保持し、後続 turn_end で turn.id に紐付ける。
            // 本文の確定は後続の turn_end が行うため、テキスト系は触らない。
            if (ev.log_message_id) {
              pendingPcLogMessageId = ev.log_message_id;
            }
          } else if (ev.type === "turn_complete") {
            // ユーザターン完了（GM/PC のレスポンス連鎖が終わり、ユーザ入力待ちへ戻った）。
            // 残った未確定吹き出しは捨てる（turn_end でほぼ消えるはず）。
            newPending.length = 0;
            setScenarioPending([]);
            // ターンが確定しないまま終わった場合（PC が本文ゼロ等）のライブスケッチも畳む。
            pendingReasoning = "";
            setScenarioStreamingReasoning(null);
            // 経過時間を記録する。同一レスポンス内の全GMバブル（複数話者ブロック）に同じ値を共有する。
            const elapsed = performance.now() - turnStartedAt;
            if (ev.turn_ids.length > 0) {
              setElapsedMap((prev) => {
                const next = { ...prev };
                for (const tid of ev.turn_ids) next[tid] = elapsed;
                return next;
              });
            }
            // レスポンス連鎖はここで終わっている。この後に走る整合性の再取得を待たずに
            // 入力欄を解放する（待つと、バブルを出し終えたあとに応答待ちインジケータが
            // 再点灯したまま数秒〜十数秒残る）。
            if (isLatestSend()) setSending(false);
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
            // GM 由来は message のみ、PC 由来は character も付く。
            ok = false;
            if (ev.character) {
              setError(`${ev.character}: ${ev.message}`);
            } else {
              setError(ev.message);
            }
            return false;
          } else if (ev.type === "done") {
            return false;
          }
        },
        onError: (e) => {
          ok = false;
          setError(String(e));
        },
      });

      // ストリーム終了時（エラー・切断含む）に残ったライブスケッチを畳む。
      setScenarioStreamingReasoning(null);

      try {
        // 完了後にサーバから真の turns を取り直して整合性確保。
        // ここは `sending` を落としたあとの裏処理なので、待っている間もユーザは
        // 次の発話を入力・送信できる。次の送信が始まっていたら着弾を捨てる。
        if (sessionId === activeSessionIdRef.current && isLatestSend()) {
          try {
            const ts = await fetchScenarioTurns(sessionId, {
              limit: TURN_PAGE_SIZE,
            });
            if (isLatestSend()) mergeTurnWindow(ts);
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
        // `sending` は通常 turn_complete で落ちているが、エラー・中断でそこへ
        // 到達しなかった場合の保険としてここでも落とす（次の送信中なら触らない）。
        if (isLatestSend()) {
          setScenarioPending([]);
          setSending(false);
        }
      }
      return { ok, firstSavedTurnId };
    },
    [
      activeScenarioSession,
      activeSessionIdRef,
      mergeTurnWindow,
      setSending,
      setError,
      setElapsedMap,
      setMsgLogIds,
    ],
  );

  /**
   * 引き直しに失敗したとき、巻き戻した枝を本線へ戻す。
   *
   * 巻き戻しは非活性化なので、元のレスポンスは枝として DB に残っている。
   * 失敗した試行で途中まで保存されたターン（ユーザ発話だけ、など）は枝として
   * 残す意味がなく、放置すると枝ナビに中身のない兄弟が増え続けるので物理削除する。
   * 失敗分は必ず最新の `turn_index` を持つので、その先頭から消せば他の枝は巻き添えにならない。
   */
  const restoreRolledBackGeneration = useCallback(
    async (
      sessionId: string,
      generationId: string,
      failedFirstTurnId: string | null,
    ) => {
      if (failedFirstTurnId) {
        await deleteScenarioTurnsFrom(sessionId, failedFirstTurnId, false);
      }
      const turns = await activateScenarioGeneration(
        sessionId,
        generationId,
        TURN_PAGE_SIZE,
      );
      // DB は元に戻すが、画面への反映は開いたままのセッションに限る
      // （待っている間に別セッションへ移っていたら、そちらの表示を壊さない）。
      if (activeSessionIdRef.current === sessionId) applyTurnWindow(turns);
    },
    [activeSessionIdRef, applyTurnWindow],
  );

  /**
   * シナリオの GM 応答を 1 レスポンス（=同一 response_key の話者ブロック群）丸ごと再生成する。
   *
   * レスポンス境界は `response_key` を共有する連続バブル列で判定する:
   *   - GM の 1 回の LLM 呼出 = 同一 response_key の GM バブル列（複数ターン=話者ブロックを含みうる）
   *   - その直前に user 発話があれば通常レスポンス → user 起点で再ストリーム
   *   - 直前に user 発話がなければ auto_advance レスポンス → GM 列の先頭から
   *     auto_advance=true で再ストリーム
   *
   * 巻き戻しは非活性化（枝として保持）なので、引き直した結果が気に入らなければ
   * 枝ナビ（◀ 2/3 ▶）で元のレスポンスへ戻せる。巻き戻し起点をユーザ発話に揃えるのは、
   * 兄弟枝の分岐点（＝その 1 つ前の turn_index）を毎回同じ値にするため。
   * 引き直しがエラーで終わった場合は `restoreRolledBackGeneration` で巻き戻しを戻す。
   */
  const handleScenarioRegenerate = useCallback(async () => {
    if (!activeScenarioSession) return;
    if (scenarioTurns.length === 0) return;

    // 末尾 GM 列の先頭 index を response_key（同一レスポンスの指紋）の連続性で探す。
    // 末尾が user の場合（GM 応答待ち状態）はそのまま user を起点にする。
    let lastTurnStart = scenarioTurns.length - 1;
    if (scenarioTurns[lastTurnStart].speaker_type !== "user") {
      const tailKey = scenarioTurns[lastTurnStart].response_key;
      while (lastTurnStart > 0) {
        const prev = scenarioTurns[lastTurnStart - 1];
        if (prev.speaker_type === "user") break;
        if (prev.response_key !== tailKey) break;
        lastTurnStart--;
      }
    }

    // 直前に user 発話があるかを見て、通常 / auto_advance を判別。
    const prev = lastTurnStart > 0 ? scenarioTurns[lastTurnStart - 1] : null;
    let pivot: ScenarioTurn;
    let resend: () => Promise<ScenarioSendResult>;
    if (prev && prev.speaker_type === "user") {
      // 通常レスポンス: user を含めて巻き戻し、同じ発話で再ストリーム
      pivot = prev;
      const content = prev.content;
      resend = () => handleScenarioSend(content, false);
    } else if (scenarioTurns[lastTurnStart].speaker_type !== "user") {
      // auto_advance レスポンス: GM 列先頭から巻き戻して auto_advance で再ストリーム
      pivot = scenarioTurns[lastTurnStart];
      resend = () => handleScenarioSend("", true);
    } else {
      // 末尾が user で GM 応答が無い特殊状態（前回ストリームエラー後など）
      pivot = scenarioTurns[lastTurnStart];
      const content = pivot.content;
      resend = () => handleScenarioSend(content, false);
    }

    const sessionId = activeScenarioSession.id;
    const pivotIndex = pivot.turn_index;
    // 引き直しに失敗したとき本線へ戻すための枝キー。枝機構より前に作られたターンは
    // null で、その場合は戻せない（従来どおりエラー表示だけで終わる）。
    const pivotGenerationId = pivot.generation_id ?? null;
    try {
      await deleteScenarioTurnsFrom(sessionId, pivot.id);
      setScenarioTurns((prevTurns) =>
        prevTurns.filter((t) => t.turn_index < pivotIndex),
      );
      const result = await resend();
      if (!result.ok && pivotGenerationId) {
        // 引き直しが失敗した。ガチャを外しただけで元のレスポンスを失わないよう、
        // 巻き戻した枝を本線へ戻す（エラー表示は resend 側が出したものを残す）。
        await restoreRolledBackGeneration(
          sessionId,
          pivotGenerationId,
          result.firstSavedTurnId,
        );
      }
    } catch (e) {
      setError(String(e));
    }
  }, [
    activeScenarioSession,
    scenarioTurns,
    handleScenarioSend,
    restoreRolledBackGeneration,
    setError,
  ]);

  /**
   * シナリオの GM 応答を 1 レスポンス分破棄してユーザ入力待ちに戻す。
   *
   * `handleScenarioRegenerate` と異なり、削除後に再ストリームしない。
   * 主な用途: ユーザが auto_advance（無入力 Enter）で GM 続きを促した結果を
   * 気に入らず、その GM 応答を捨てて自分で発話を入力したい場合。
   *
   * 削除対象は末尾 GM 列のみ（同一 response_key のバブル列）。
   * 直前のユーザ発話があれば残す（そこから次の発話を入力できる）。
   * 末尾が user の状態（GM 未応答）では何もしない。
   */
  const handleScenarioDiscard = useCallback(async () => {
    if (!activeScenarioSession) return;
    if (scenarioTurns.length === 0) return;

    const lastIndex = scenarioTurns.length - 1;
    if (scenarioTurns[lastIndex].speaker_type === "user") return;

    // 末尾 GM 列の先頭を response_key（同一レスポンスの指紋）の連続性で探す
    let groupStart = lastIndex;
    const tailKey = scenarioTurns[lastIndex].response_key;
    while (groupStart > 0) {
      const prev = scenarioTurns[groupStart - 1];
      if (prev.speaker_type === "user") break;
      if (prev.response_key !== tailKey) break;
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
   *
   * ここだけ枝を残さず物理削除する（`keepVariants=false`）。発言そのものを
   * 書き換える以上、その発言に対して引いた過去のガチャはすべて無効だからで、
   * 内容の違う発話が同じ枝リストに並ぶのも防げる。
   */
  const handleScenarioEditUserTurn = useCallback(
    async (turnId: string, newContent: string) => {
      if (!activeScenarioSession) return;
      const target = scenarioTurns.find((t) => t.id === turnId);
      if (!target) return;
      try {
        await deleteScenarioTurnsFrom(activeScenarioSession.id, turnId, false);
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

  /**
   * 末尾ユーザ発話の削除。
   *
   * `handleScenarioEditUserTurn` と違い、削除するだけで再ストリームしない
   * （送ってしまった発話をなかったことにして、入力からやり直すための操作）。
   * 削除 API は「指定ターン以降」しか持たないため、呼び出し側は末尾のユーザ発話に
   * だけ使うこと。UI 上もセッション末尾のバブルにしかゴミ箱を出さない。
   *
   * 編集と同じく枝は残さない（`keepVariants=false`）。発話そのものが消える以上、
   * その発話に対して引いた過去のガチャは行き場が無いため。
   */
  const handleScenarioDeleteUserTurn = useCallback(
    async (turnId: string) => {
      if (!activeScenarioSession) return;
      const target = scenarioTurns.find((t) => t.id === turnId);
      if (!target) return;
      try {
        await deleteScenarioTurnsFrom(activeScenarioSession.id, turnId, false);
        setScenarioTurns((prev) =>
          prev.filter((t) => t.turn_index < target.turn_index),
        );
      } catch (e) {
        setError(String(e));
      }
    },
    [activeScenarioSession, scenarioTurns, setError],
  );

  /**
   * 枝（レスポンスガチャ）を切り替える。
   *
   * 対象が過去のレスポンスなら、その分岐点より後の本線はサーバ側で巻き戻される
   * （下流は復元しない）ため、呼び出し側で確認を取ってから使うこと。
   */
  const handleScenarioSwitchVariant = useCallback(
    async (generationId: string) => {
      if (!activeScenarioSession) return;
      try {
        const turns = await activateScenarioGeneration(
          activeScenarioSession.id,
          generationId,
          TURN_PAGE_SIZE,
        );
        applyTurnWindow(turns);
      } catch (e) {
        setError(String(e));
      }
    },
    [activeScenarioSession, applyTurnWindow, setError],
  );

  /**
   * GM / PC / NPC / Narrator の発話をユーザの手で上書きする。
   *
   * 枝は生やさず、その場で本文だけを差し替える（先の展開はそのまま残る）。
   * ユーザ発話の編集（`handleScenarioEditUserTurn`）とは別物。
   */
  const handleScenarioEditResponse = useCallback(
    async (turnId: string, newContent: string) => {
      if (!activeScenarioSession) return;
      try {
        const updated = await patchScenarioTurn(
          activeScenarioSession.id,
          turnId,
          newContent,
        );
        setScenarioTurns((prev) =>
          prev.map((t) => (t.id === turnId ? updated : t)),
        );
      } catch (e) {
        setError(String(e));
      }
    },
    [activeScenarioSession, setError],
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
   * 内部は `handleScenarioSend("", autoAdvance=true, target)` のラッパー。
   * target に PC枠名を渡せばその PC、"GM" なら GM、"ALL" ならランダム PC へルーティングされる。
   */
  const handleScenarioYieldTo = useCallback(
    async (target: string) => {
      await handleScenarioSend("", true, target);
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
    hasOlderTurns,
    loadingOlderTurns,
    loadOlderScenarioTurns,
    scenarioPending,
    scenarioStreamingReasoning,
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
    handleScenarioDeleteUserTurn,
    handleScenarioSwitchVariant,
    handleScenarioEditResponse,
    handleSynopsisChange,
    handleSynopsisCreate,
    handleOpenSynopsisCreate,
    handleCancelSynopsisCreate,
  };
}
