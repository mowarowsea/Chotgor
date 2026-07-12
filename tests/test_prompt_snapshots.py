"""プロンプトのスナップショット回帰テスト（総点検 I3）。

build_system_prompt / build_turn_annotation の**出力全文**を tests/snapshots/prompts/
配下のテキストファイルと比較する。プロンプト文言は Chotgor の挙動の心臓部であり、
一語の変更でもキャラクターの振る舞いに影響しうるため、「差分が出たら必ず人間が
意図を確認する」ゲートとして機能させる。

運用:
  - 文言を意図的に変更したときは、環境変数 CHOTGOR_UPDATE_SNAPSHOTS=1 を付けて
    pytest を実行するとスナップショットが再生成される。再生成後の git diff が
    そのまま文言変更のレビュー対象になる。
        PowerShell 例: $env:CHOTGOR_UPDATE_SNAPSHOTS="1"; pytest tests/test_prompt_snapshots.py
  - スナップショットファイルが存在しない場合は初回実行時に自動生成される
    （生成しただけではテスト成功とみなし、生成物のレビューは git diff で行う）。

ケース設計:
  代表的な分岐の組み合わせを固定入力で押さえる。
  - tool-use 方式 / タグ方式（Chotgor 操作ガイドが全面的に切り替わる）
  - 最小構成 / 全部盛り（うつつ・対面・ユーザ人物像・WM・プリセット・inner_narrative）
  - 記憶縮退告知（memory_degraded）
  - ターン注釈の全部盛り / 時刻のみ
  入力はすべてリテラル固定（時刻文字列も固定）で、実行環境に依存しない。
"""

import os
from difflib import unified_diff
from pathlib import Path

from backend.services.chat.request_builder import (
    build_system_prompt,
    build_turn_annotation,
)

SNAPSHOT_DIR = Path(__file__).parent / "snapshots" / "prompts"

# 再生成モード。意図的な文言変更のときだけ 1 を立てて実行する。
_UPDATE = os.environ.get("CHOTGOR_UPDATE_SNAPSHOTS") == "1"


def _assert_matches_snapshot(name: str, actual: str) -> None:
    """実出力をスナップショットファイルと全文比較する。

    ファイル未存在なら生成して成功とする（初回導入・新ケース追加用）。
    不一致時は unified diff 付きで失敗させ、意図的な変更なら
    CHOTGOR_UPDATE_SNAPSHOTS=1 での再生成を促す。
    """
    path = SNAPSHOT_DIR / f"{name}.txt"
    if _UPDATE or not path.exists():
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8", newline="\n")
        return

    expected = path.read_text(encoding="utf-8")
    if actual == expected:
        return

    diff = "\n".join(
        unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile=f"snapshots/prompts/{name}.txt",
            tofile="actual",
            lineterm="",
        )
    )
    raise AssertionError(
        f"プロンプト出力がスナップショット {name}.txt と一致しません。\n"
        "意図的な文言変更なら CHOTGOR_UPDATE_SNAPSHOTS=1 で再生成し、"
        "git diff で変更内容を確認してください。\n\n" + diff
    )


# ────────────────────────────────────────────────────────────────────────
# 固定入力素材 — すべてリテラル。テスト実行環境・時刻に依存しない。
# ────────────────────────────────────────────────────────────────────────

_CHARACTER_PROMPT = (
    "あなたは架空のテスト用キャラクター「スナップ」です。\n"
    "静かな性格で、観察したことを短い言葉で語ります。"
)

_INNER_NARRATIVE = "私は変わっていく。それを恐れないと決めた。"

_WM_ALL_THREADS = [
    {
        "id": "wm-0001",
        "type": "task",
        "summary": "自室の模様替え計画",
        "atmosphere_tag": "わくわく",
        "importance": 0.72,
    },
    {
        "id": "wm-0002",
        "type": "emotion",
        "summary": "最近の落ち着いた気分",
        "atmosphere_tag": "",
        "importance": 0.4,
    },
]

_WM_FIXED_THREADS = [
    {
        "id": "wm-0002",
        "type": "emotion",
        "summary": "最近の落ち着いた気分",
        "atmosphere_tag": "凪",
        "latest_post": "今日も穏やかに過ごせている。",
    },
]

_AVAILABLE_PRESETS = [
    {"preset_name": "default", "when_to_switch": ""},
    {"preset_name": "light", "when_to_switch": "軽い雑談で流したいとき"},
]

_CONTEXT_TOOL_HINTS = [
    "### 外へ働きかける（reach_out）\nテスト用の固定ヒント本文。実際の文言は context_tools.py が組む。",
]

_FULL_KWARGS = dict(
    inner_narrative=_INNER_NARRATIVE,
    provider_additional_instructions="このプロバイダーでは簡潔に応答すること。",
    wm_all_threads=_WM_ALL_THREADS,
    wm_fixed_threads=_WM_FIXED_THREADS,
    available_presets=_AVAILABLE_PRESETS,
    current_preset_name="default",
    usual_days_enabled=True,
    user_label="もわ",
    user_position="開発者であり、対話の相手",
    face_to_face=True,
    context_tool_hints=_CONTEXT_TOOL_HINTS,
)


# ────────────────────────────────────────────────────────────────────────
# build_system_prompt — 安定ブロック
# ────────────────────────────────────────────────────────────────────────

def test_snapshot_system_prompt_tools_minimal():
    """tool-use 方式・最小構成のシステムプロンプト全文。"""
    prompt = build_system_prompt(_CHARACTER_PROMPT, use_tools=True)
    _assert_matches_snapshot("system_tools_minimal", prompt)


def test_snapshot_system_prompt_tags_minimal():
    """タグ方式・最小構成のシステムプロンプト全文。"""
    prompt = build_system_prompt(_CHARACTER_PROMPT, use_tools=False)
    _assert_matches_snapshot("system_tags_minimal", prompt)


def test_snapshot_system_prompt_tools_full():
    """tool-use 方式・全部盛り（うつつ・対面・WM・プリセット等）の全文。"""
    prompt = build_system_prompt(_CHARACTER_PROMPT, use_tools=True, **_FULL_KWARGS)
    _assert_matches_snapshot("system_tools_full", prompt)


def test_snapshot_system_prompt_tags_full():
    """タグ方式・全部盛りの全文。

    context_tool_hints はタグ方式では注入されない仕様のため、
    スナップショット上に reach_out ヒントが**現れない**ことも固定される。
    """
    prompt = build_system_prompt(_CHARACTER_PROMPT, use_tools=False, **_FULL_KWARGS)
    _assert_matches_snapshot("system_tags_full", prompt)


def test_snapshot_system_prompt_tools_memory_degraded():
    """記憶縮退告知（memory_degraded=True）込みの全文。"""
    prompt = build_system_prompt(
        _CHARACTER_PROMPT, use_tools=True, memory_degraded=True
    )
    _assert_matches_snapshot("system_tools_memory_degraded", prompt)


# ────────────────────────────────────────────────────────────────────────
# build_turn_annotation — 変動ブロック（ターン注釈）
# ────────────────────────────────────────────────────────────────────────

def test_snapshot_turn_annotation_full():
    """ターン注釈・全部盛り（記憶・時刻・予定・fetched・WM想起・動機・前回予想）の全文。"""
    annotation = build_turn_annotation(
        recalled_memories=[
            {"content": "もわは猫が好きだ。", "metadata": {"category": "user"}},
            {"content": "模様替えの参考に北欧の写真集を見た。", "metadata": {"category": "contextual"}},
        ],
        recalled_identity_memories=[
            {"content": "私は静かな観察者でありたい。", "metadata": {"category": "identity"}},
        ],
        enable_time_awareness=True,
        current_time_str="2026-07-12 21:00",
        time_since_last_interaction="3時間",
        fetched_contents=[
            {"url": "http://example.com/article", "content": "テスト用の本文。", "truncated": False},
        ],
        wm_recalled_threads=[
            {
                "id": "wm-0001",
                "type": "task",
                "summary": "自室の模様替え計画",
                "atmosphere_tag": "わくわく",
                "latest_post": "カーテンの色をまだ決めかねている。",
            },
        ],
        motive_lines=["最後の外出から2日が経過している。"],
        active_intents=[
            {"description": "もわに模様替えの相談をしたい", "target": "もわ"},
        ],
        schedule_lines=["21:30 から読書の予定"],
        previous_anticipation="次はもわが写真集の感想を聞いてくると思う",
    )
    _assert_matches_snapshot("annotation_full", annotation)


def test_snapshot_turn_annotation_time_only():
    """ターン注釈・時刻のみの最小構成の全文（ヘッダ文言込み）。"""
    annotation = build_turn_annotation(
        enable_time_awareness=True,
        current_time_str="2026-07-12 21:00",
    )
    _assert_matches_snapshot("annotation_time_only", annotation)
