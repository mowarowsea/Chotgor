/**
 * 新規セッション作成モーダル。
 *
 * セッション種別（1on1 / グループ / シナリオ）をタブで選び、種別ごとの設定を入力して作成する。
 */
import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import type { Character, Model, ScenarioTemplate } from "../api";
import { charNameOf, presetNameOf, fetchScenarioTemplates } from "../api";
import { CharacterAvatar } from "./ChatBubbles";

/** セッション種別。 */
type SessionType = "1on1" | "group" | "scenario";

interface Props {
  /** 利用可能なモデル一覧（"{char}@{preset}" 形式の id を持つ）。 */
  models: Model[];
  /** キャラクター一覧（Afterglow デフォルト値の参照に使う）。 */
  characters: Character[];
  /** モーダルを閉じるコールバック。 */
  onClose: () => void;
  /** 1on1 チャット作成コールバック。 */
  onNewChat: (modelId: string, afterglow: boolean) => void;
  /** グループチャット作成コールバック（司会モデルはシステム設定で管理するため引数に含まない）。 */
  onNewGroupChat: (participants: string[], maxAutoTurns: number) => void;
  /** シナリオ起動コールバック。 */
  onStartScenario: (scenarioId: string, title?: string) => void;
}

/** 種別タブの定義。 */
const TYPE_TABS: { key: SessionType; label: string; desc: string }[] = [
  { key: "1on1", label: "1on1", desc: "1人のキャラと会話" },
  { key: "group", label: "Group", desc: "複数キャラ + 司会AI" },
  { key: "scenario", label: "Scenario", desc: "ナレーター進行の物語" },
];

/** セクション見出しの共通スタイル。 */
function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="text-[10px] text-ch-t3 font-mono mb-1.5"
      style={{ letterSpacing: "0.06em" }}
    >
      {children}
    </div>
  );
}

/** 新規セッション作成モーダル本体。 */
export default function NewSessionPicker({
  models,
  characters,
  onClose,
  onNewChat,
  onNewGroupChat,
  onStartScenario,
}: Props) {
  const [type, setType] = useState<SessionType>("1on1");

  /** モデル一覧をキャラクター名 → プリセット名配列にグルーピングする。 */
  const charMap = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const m of models) {
      const c = charNameOf(m.id);
      const p = presetNameOf(m.id);
      if (!map.has(c)) map.set(c, []);
      map.get(c)!.push(p);
    }
    return map;
  }, [models]);
  const charNames = useMemo(() => [...charMap.keys()], [charMap]);

  /* ── 1on1 用 state ── */
  const [selChar, setSelChar] = useState("");
  const [selPreset, setSelPreset] = useState("");
  const [afterglow, setAfterglow] = useState(false);

  /* ── group 用 state ── */
  const [groupSelected, setGroupSelected] = useState<Set<string>>(new Set());
  const [maxAutoTurns, setMaxAutoTurns] = useState(3);

  /* ── scenario 用 state ── */
  const [templates, setTemplates] = useState<ScenarioTemplate[]>([]);
  const [scId, setScId] = useState("");
  const [scTitle, setScTitle] = useState("");

  // 初期選択キャラクターを設定する。
  useEffect(() => {
    if (!selChar && charNames.length > 0) setSelChar(charNames[0]);
  }, [charNames, selChar]);

  // 選択キャラのプリセット候補。選択中プリセットが候補外なら先頭へ寄せる。
  const availablePresets = selChar ? charMap.get(selChar) ?? [] : [];
  useEffect(() => {
    if (availablePresets.length > 0 && !availablePresets.includes(selPreset)) {
      setSelPreset(availablePresets[0]);
    }
  }, [selChar, availablePresets, selPreset]);

  // 選択キャラの Afterglow デフォルト値を反映する。
  useEffect(() => {
    const char = characters.find((c) => c.name === selChar);
    setAfterglow(char?.afterglow_default ?? false);
  }, [selChar, characters]);

  // シナリオタブを開いたらテンプレート一覧を取得する。
  useEffect(() => {
    if (type === "scenario") {
      fetchScenarioTemplates().then(setTemplates).catch(() => setTemplates([]));
    }
  }, [type]);

  // Esc キーで閉じる。
  useEffect(() => {
    const fn = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", fn);
    return () => document.removeEventListener("keydown", fn);
  }, [onClose]);

  /** グループ参加者の選択をトグルする。 */
  const toggleGroupModel = (modelId: string) => {
    setGroupSelected((prev) => {
      const next = new Set(prev);
      if (next.has(modelId)) next.delete(modelId);
      else next.add(modelId);
      return next;
    });
  };

  /** 現在の入力で作成可能かどうか。 */
  const canCreate =
    type === "1on1"
      ? !!selChar && !!selPreset
      : type === "group"
        ? groupSelected.size >= 2
        : !!scId;

  /** 作成を確定する。 */
  const handleCreate = () => {
    if (!canCreate) return;
    if (type === "1on1") {
      onNewChat(`${selChar}@${selPreset}`, afterglow);
    } else if (type === "group") {
      onNewGroupChat([...groupSelected], maxAutoTurns);
    } else {
      onStartScenario(scId, scTitle.trim() || undefined);
    }
    onClose();
  };

  // モーダルは createPortal で document.body 直下へ出す。
  // サイドバー（transform を持つ）の内側でレンダリングすると position:fixed が
  // サイドバー基準になり、画面中央ではなくサイドバー幅に固定されてしまうため。
  return createPortal(
    <div
      onClick={onClose}
      className="fixed inset-0 z-[100] flex items-center justify-center p-4"
      style={{ background: "var(--ch-overlay)" }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-[440px] bg-ch-bg rounded-xl flex flex-col overflow-hidden"
        style={{ border: "1px solid var(--ch-sep2)", boxShadow: "var(--ch-shadow)", maxHeight: "85vh" }}
      >
        {/* ヘッダー */}
        <div
          className="flex items-center px-4 py-2.5 shrink-0"
          style={{ borderBottom: "1px solid var(--ch-sep)" }}
        >
          <span className="text-ch-t1 text-xs font-semibold flex-1">新しい会話</span>
          <button onClick={onClose} className="text-ch-t3 hover:text-ch-t1 text-base px-1.5">
            ✕
          </button>
        </div>

        {/* ボディ */}
        <div className="px-4 py-4 overflow-y-auto flex flex-col gap-4">
          {/* 種別タブ */}
          <div>
            <SectionLabel>TYPE</SectionLabel>
            <div className="grid grid-cols-3 gap-1.5">
              {TYPE_TABS.map((t) => {
                const active = type === t.key;
                return (
                  <button
                    key={t.key}
                    onClick={() => setType(t.key)}
                    className="text-left rounded-lg px-2.5 py-2 transition-colors"
                    style={{
                      border: `1px solid ${active ? "var(--ch-accent)" : "var(--ch-sep2)"}`,
                      background: active ? "oklch(50% 0.13 226 / 0.08)" : "transparent",
                    }}
                  >
                    <div
                      className="text-xs font-semibold mb-0.5"
                      style={{ color: active ? "var(--ch-accent)" : "rgb(var(--ch-t1))" }}
                    >
                      {t.label}
                    </div>
                    <div className="text-[10px] text-ch-t3 leading-tight">{t.desc}</div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* 1on1 設定 */}
          {type === "1on1" && (
            <>
              <div>
                <SectionLabel>CHARACTER</SectionLabel>
                {charNames.length === 0 ? (
                  <p className="text-ch-t3 text-xs">利用可能なキャラクターがありません</p>
                ) : (
                  <div className="flex gap-1.5 flex-wrap">
                    {charNames.map((c) => {
                      const active = selChar === c;
                      return (
                        <button
                          key={c}
                          onClick={() => setSelChar(c)}
                          className="flex items-center gap-1.5 rounded-md px-2 py-1 text-xs transition-colors"
                          style={{
                            border: `1px solid ${active ? "var(--ch-accent)" : "var(--ch-sep2)"}`,
                            background: active ? "oklch(50% 0.13 226 / 0.08)" : "transparent",
                            color: active ? "var(--ch-accent)" : "rgb(var(--ch-t2))",
                          }}
                        >
                          <CharacterAvatar characterName={c} size={18} />
                          {c}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
              {availablePresets.length > 0 && (
                <div>
                  <SectionLabel>PRESET</SectionLabel>
                  <div className="flex gap-1.5 flex-wrap">
                    {availablePresets.map((p) => {
                      const active = selPreset === p;
                      return (
                        <button
                          key={p}
                          onClick={() => setSelPreset(p)}
                          className="rounded-md px-2.5 py-1 text-xs transition-colors"
                          style={{
                            border: `1px solid ${active ? "var(--ch-accent)" : "var(--ch-sep2)"}`,
                            background: active ? "oklch(50% 0.13 226 / 0.10)" : "transparent",
                            color: active ? "var(--ch-accent)" : "rgb(var(--ch-t2))",
                          }}
                        >
                          {p}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
              <label className="flex items-start gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={afterglow}
                  onChange={(e) => setAfterglow(e.target.checked)}
                  className="mt-0.5 shrink-0 accent-ch-accent"
                />
                <span className="text-ch-t2 text-xs leading-snug">
                  前回の流れを引き継ぐ
                  <span className="block text-ch-t3 mt-0.5">
                    直近5ターンの会話を前置きとして引き継ぎます
                  </span>
                </span>
              </label>
            </>
          )}

          {/* グループ設定 */}
          {type === "group" && (
            <>
              <div>
                <SectionLabel>CHARACTERS · 2名以上選択</SectionLabel>
                {models.length === 0 ? (
                  <p className="text-ch-t3 text-xs">利用可能なモデルがありません</p>
                ) : (
                  <div className="flex flex-col gap-1 max-h-40 overflow-y-auto">
                    {models.map((m) => {
                      const sel = groupSelected.has(m.id);
                      return (
                        <label key={m.id} className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={sel}
                            onChange={() => toggleGroupModel(m.id)}
                            className="accent-ch-accent"
                          />
                          <CharacterAvatar characterName={charNameOf(m.id)} size={18} />
                          <span className="text-ch-t2 text-xs truncate">
                            {charNameOf(m.id)}
                            <span className="text-ch-t3 font-mono ml-1">@{presetNameOf(m.id)}</span>
                          </span>
                        </label>
                      );
                    })}
                  </div>
                )}
              </div>
              <div>
                <SectionLabel>
                  最大自動ターン数: <span className="text-ch-t1">{maxAutoTurns}</span>
                  {maxAutoTurns >= 5 && <span className="text-amber-600 ml-1">⚠ API消費増</span>}
                </SectionLabel>
                <input
                  type="range"
                  min={1}
                  max={10}
                  value={maxAutoTurns}
                  onChange={(e) => setMaxAutoTurns(Number(e.target.value))}
                  className="w-full accent-ch-accent"
                />
              </div>
            </>
          )}

          {/* シナリオ設定 */}
          {type === "scenario" && (
            <>
              <div>
                <SectionLabel>SCENARIO</SectionLabel>
                {templates.length === 0 ? (
                  <p className="text-ch-t3 text-xs">
                    シナリオがありません。
                    <a
                      href="/ui/scenarios/new"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="underline ml-1 hover:text-ch-t2"
                    >
                      backend で登録
                    </a>
                  </p>
                ) : (
                  <div className="flex flex-col gap-1.5">
                    {templates.map((t) => {
                      const active = scId === t.id;
                      return (
                        <button
                          key={t.id}
                          onClick={() => setScId(t.id)}
                          className="text-left rounded-lg px-2.5 py-2 transition-colors"
                          style={{
                            border: `1px solid ${active ? "var(--ch-accent)" : "var(--ch-sep2)"}`,
                            background: active ? "oklch(50% 0.13 226 / 0.08)" : "transparent",
                          }}
                        >
                          <div
                            className="text-xs font-semibold"
                            style={{ color: active ? "var(--ch-accent)" : "rgb(var(--ch-t1))" }}
                          >
                            {t.title}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
              <div>
                <SectionLabel>セッションタイトル（任意）</SectionLabel>
                <input
                  value={scTitle}
                  onChange={(e) => setScTitle(e.target.value)}
                  placeholder="空欄ならシナリオ名をコピー"
                  className="w-full bg-ch-s3 text-ch-t1 placeholder-ch-t3 text-xs rounded-md px-2 py-1.5 focus:outline-none"
                  style={{ border: "1px solid var(--ch-sep2)" }}
                />
              </div>
            </>
          )}
        </div>

        {/* フッター */}
        <div
          className="flex justify-end gap-2 px-4 py-2.5 shrink-0"
          style={{ borderTop: "1px solid var(--ch-sep)" }}
        >
          <button
            onClick={onClose}
            className="text-ch-t2 hover:text-ch-t1 text-xs px-3.5 py-1.5 rounded-md transition-colors"
          >
            キャンセル
          </button>
          <button
            onClick={handleCreate}
            disabled={!canCreate}
            className="text-white text-xs font-medium px-3.5 py-1.5 rounded-md transition-opacity disabled:opacity-30 disabled:cursor-not-allowed"
            style={{ background: "var(--ch-accent)" }}
          >
            作成
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
