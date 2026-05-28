/**
 * あらすじ作成モーダル。
 *
 * ターン完了後にあらすじ作成の閾値へ達したとき自動表示される。また、左上の
 * シナリオ設定モーダルの「自動作成」ボタンや、チャット下部のバナーからも開ける。
 *
 * 旧設計はあらすじ蒸留をターン開始前に同期実行していたため「謎の遅延」「あらすじ
 * モデル切り替え忘れによる大量トークン消費」を招いていた。本モーダルは作成を
 * ユーザの明示操作に切り替え、その場で蒸留用 Preset を選ばせることでそれを防ぐ。
 * 「作成」押下後は裏で蒸留が走り、ユーザはそのままチャットを続けられる。
 */
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import type { ScenarioPreset } from "../api";

interface Props {
  /** 選択候補となる LLM プリセット一覧（fetchScenarioPresets の結果）。 */
  presets: ScenarioPreset[];
  /** 初期選択するあらすじ蒸留プリセット ID（セッション既定）。 */
  currentSynopsisPresetId: string;
  /** 「作成」押下時のコールバック。選んだ preset ID を渡す。 */
  onCreate: (presetId: string) => void;
  /** キャンセル（バナーへ降格）コールバック。背景クリック・Esc・キャンセルボタン共通。 */
  onCancel: () => void;
}

/** あらすじ作成モーダル本体。 */
export default function SynopsisCreateModal({
  presets,
  currentSynopsisPresetId,
  onCreate,
  onCancel,
}: Props) {
  // 選択中の preset。初期値はセッション既定（記憶済みのあらすじモデル）。
  const [selectedId, setSelectedId] = useState(currentSynopsisPresetId);

  // セッション側で外部変更があったら選択も追従させる。
  useEffect(() => {
    setSelectedId(currentSynopsisPresetId);
  }, [currentSynopsisPresetId]);

  // Esc キーでキャンセル（バナーへ降格）。
  useEffect(() => {
    const fn = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    document.addEventListener("keydown", fn);
    return () => document.removeEventListener("keydown", fn);
  }, [onCancel]);

  const create = () => {
    if (!selectedId) return;
    onCreate(selectedId);
  };

  return createPortal(
    <div
      onClick={onCancel}
      className="fixed inset-0 z-[100] flex items-center justify-center p-4"
      style={{ background: "var(--ch-overlay)" }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-[480px] bg-ch-bg rounded-xl flex flex-col overflow-hidden"
        style={{
          border: "1px solid var(--ch-sep2)",
          boxShadow: "var(--ch-shadow)",
          maxHeight: "85vh",
        }}
      >
        {/* ヘッダー */}
        <div
          className="flex items-center px-4 py-2.5 shrink-0"
          style={{ borderBottom: "1px solid var(--ch-sep)" }}
        >
          <span className="text-ch-t1 text-xs font-semibold flex-1">
            あらすじを作成
          </span>
          <button
            onClick={onCancel}
            className="text-ch-t3 hover:text-ch-t1 text-base px-1.5"
          >
            ✕
          </button>
        </div>

        {/* ボディ */}
        <div className="px-4 py-4 overflow-y-auto flex flex-col gap-4">
          <p className="text-ch-t3 text-xs leading-relaxed">
            これまでの履歴を要約して「あらすじ」に統合します。蒸留に使うモデルを選んで
            「作成」を押すと裏で実行され、そのままチャットを続けられます。
          </p>

          <div className="flex flex-col gap-1.5">
            <div className="text-[11px] text-ch-t3">
              あらすじ作成用モデル（軽量モデルで節約可。選択は次回以降の既定にもなる）
            </div>
            {presets.length === 0 ? (
              <p className="text-ch-t3 text-xs">
                LLM プリセットがありません。
                <a
                  href="/ui/presets"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline ml-1 hover:text-ch-t2"
                >
                  backend で登録
                </a>
              </p>
            ) : (
              <div className="flex gap-1.5 flex-wrap">
                {presets.map((p) => {
                  const active = selectedId === p.id;
                  return (
                    <button
                      key={p.id}
                      onClick={() => setSelectedId(p.id)}
                      className="rounded-md px-2.5 py-1 text-xs transition-colors"
                      style={{
                        border: `1px solid ${active ? "var(--ch-accent)" : "var(--ch-sep2)"}`,
                        background: active
                          ? "oklch(50% 0.13 226 / 0.10)"
                          : "transparent",
                        color: active ? "var(--ch-accent)" : "rgb(var(--ch-t2))",
                      }}
                      title={`${p.provider} / ${p.model_id || "default"}`}
                    >
                      {p.name}
                      <span className="text-ch-t4 ml-1 font-mono text-[10px]">
                        {p.provider}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* アクション */}
          <div className="flex justify-end gap-2 pt-1">
            <button
              onClick={onCancel}
              className="text-ch-t3 hover:text-ch-t1 text-xs px-3 py-1.5 rounded-md transition-colors"
            >
              キャンセル
            </button>
            <button
              onClick={create}
              disabled={!selectedId}
              className="text-white text-xs font-medium px-3.5 py-1.5 rounded-md transition-opacity disabled:opacity-30 disabled:cursor-not-allowed"
              style={{ background: "var(--ch-accent)" }}
            >
              作成
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
