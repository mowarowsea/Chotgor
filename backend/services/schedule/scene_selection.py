"""うつつシーンの②導出 — 実現層エントリからシーン起動を選出する純関数（Phase 4・§8）。

現在ユーザが `usual_config.slots` で手動設定しているシーン起動タイミングを、
**②はる固定予定（および①世界固定）から導出**する。②の中身（「日曜はゲーム」「調べもの」
「片付け」）が、そのままうつつシーンの題材でありトリガーになる。

選出は純関数（LLM 不使用・決定論乱数 seed=キャラ+日）— schedule_plan.md §8 の裁定:
    - 対象: その日の **offline 以外**の planned エントリ（①②を問わない — 仕事の場面も生活のうち）
    - 枠（usual_scenes_per_day）の **50% を占有圧上位**から、**残りをランダム**に選出
    - シーン起動時刻はエントリ開始＋決定論ジッター
    - 既存の日次コストガード `usual_days_daily_cap` はスケジューラ側で共有

「どの予定を具体的に経験するか」は演出の抽選＝世界の物理であり、はるの裁定事項にしない。
"""

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta

# シーン起動時刻に載せる決定論ジッターの最大幅（分）。エントリ開始ちょうどに毎回始めると
# 機械的なので、行動権・escrow と同じ思想で「いつ始めるか」だけを世界乱数で揺らす。
_SCENE_JITTER_MAX_MINUTES = 45


@dataclass
class SceneSlot:
    """②導出で選ばれたうつつシーン1件 — スケジューラが起動する素材。

    Attributes:
        entry_id: 由来の schedule_entries.id（冪等キーの一部・題材の出所）。
        label: シーンの題材（エントリのラベル。GM への framing に使う）。
        fire_at: シーン起動時刻（エントリ開始＋決定論ジッター）。
        occupancy: 由来エントリの占有圧（ログ・観測用）。
        state: 由来エントリの配達状態（OnTime / active / busy）。offline は選出対象外。
    """

    entry_id: str
    label: str
    fire_at: datetime
    occupancy: float
    state: str


def _day_span(day: date) -> tuple[datetime, datetime]:
    """日付の [00:00, 翌00:00) を datetime で返す（当日エントリの抽出窓）。"""
    start = datetime(day.year, day.month, day.day)
    return start, start + timedelta(days=1)


def _scene_jitter_minutes(character_id: str, entry_id: str) -> int:
    """シーン起動ジッターを決定論導出する（乱数は世界に置く）。

    同じキャラ・同じエントリなら常に同じ待ち時間。「いつ始めるか」だけを揺らし、
    「何を経験するか」の中身には関与しない（それはシーン内で本人が決める）。

    Args:
        character_id: キャラクター ID（シードの一部）。
        entry_id: 由来エントリ ID（シードの一部）。

    Returns:
        0〜_SCENE_JITTER_MAX_MINUTES 分。
    """
    rng = random.Random(f"usual-scene-jitter:{character_id}:{entry_id}")
    return rng.randint(0, _SCENE_JITTER_MAX_MINUTES)


def select_daily_scenes(
    entries: list,
    *,
    character_id: str,
    day: date,
    scenes_per_day: int,
) -> list[SceneSlot]:
    """その日の planned エントリからうつつシーンを選出する（§8・決定論純関数）。

    毎分のスケジューラから呼ばれても結果がぶれないよう決定論乱数（seed=キャラ+日）で回す。
    選出規則:
        1. offline 以外・その日に始まるエントリを対象にする。
        2. 対象数が枠以下なら全採用。
        3. 枠を超えるなら、枠の 50%（切り上げ）を占有圧上位から、残りを決定論シャッフルで
           ランダムに選ぶ（占有圧上位を優先しつつ、単調にならないよう抽選を混ぜる）。
        4. 各シーンの起動時刻 = エントリ開始 ＋ 決定論ジッター。起動時刻の昇順で返す。

    Args:
        entries: その日を含む planned な ScheduleEntry のリスト（呼び出し側が期間で取得）。
        character_id: 対象キャラクター（決定論乱数のシード）。
        day: 対象日（この日に始まるエントリだけを候補にする）。
        scenes_per_day: 1日のシーン回数（ユーザ設定）。0 以下なら空リスト。

    Returns:
        起動時刻昇順の SceneSlot リスト。候補ゼロ・scenes_per_day<=0 なら空リスト。
    """
    if scenes_per_day <= 0:
        return []
    day_start, day_end = _day_span(day)
    # offline は「意識がない」時間なので体験の題材にならない（§8）。その日に始まるものだけ。
    eligible = [
        e for e in entries
        if str(getattr(e, "state", "") or "") != "offline"
        and getattr(e, "start_at", None) is not None
        and day_start <= e.start_at < day_end
    ]
    if not eligible:
        return []

    if len(eligible) <= scenes_per_day:
        chosen = list(eligible)
    else:
        # 占有圧降順（同値は entry_id で決定論タイブレーク）。枠の半分を上位から。
        by_occ = sorted(
            eligible,
            key=lambda e: (-float(getattr(e, "occupancy", 0.0) or 0.0), str(e.id)),
        )
        top_count = (scenes_per_day + 1) // 2  # 切り上げ — 上位を優先
        top = by_occ[:top_count]
        # 残りは決定論シャッフルでランダム抽選（seed=キャラ+日）
        rest = by_occ[top_count:]
        rng = random.Random(f"usual-scene-select:{character_id}:{day.isoformat()}")
        rng.shuffle(rest)
        chosen = top + rest[: scenes_per_day - top_count]

    slots = [
        SceneSlot(
            entry_id=str(e.id),
            label=str(getattr(e, "label", None) or "日常のひとコマ"),
            fire_at=e.start_at + timedelta(
                minutes=_scene_jitter_minutes(character_id, str(e.id))
            ),
            occupancy=float(getattr(e, "occupancy", 0.0) or 0.0),
            state=str(getattr(e, "state", "") or ""),
        )
        for e in chosen
    ]
    slots.sort(key=lambda s: s.fire_at)
    return slots


def format_scene_framing(character_name: str, label: str) -> str:
    """②導出シーンの題材を GM 向けの framing OOC に整形する。

    エントリのラベル（本人が立てた予定・世界の予定）を「今日のこの場面のきっかけ」として
    GM に渡す。中身の演出は GM の即興に委ねる（世界＝GM が外的フレームを与える思想）。

    Args:
        character_name: 主人公キャラ名。
        label: シーンの題材（エントリのラベル）。

    Returns:
        GM へ渡す OOC 文字列。label が空なら空文字列。
    """
    label = (label or "").strip()
    if not label:
        return ""
    name = (character_name or "").strip() or "本人"
    return (
        f"[OOC] この場面のきっかけ: {name} は今日「{label}」という予定・過ごし方の中にいる。"
        f"その予定にまつわる場面として幕を開けてよい（具体的な中身は即興で。"
        f"{name} の内面・選択には踏み込まない）。"
    )
