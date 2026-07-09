"""圧力エンジン — 3変数（社会圧・退屈圧・体調圧）の純関数計算。

すべてタイムライン封筒（timeline_events）と体質プロファイル
（characters.pressure_profile）からの導出で、状態を保存しない
（docs/planned/aliveness_plan.md §4.1）。LLM 不使用。

設計原則:
    - 乱数は世界に置き意志に置かない（リズム成分は character_id シードの決定論導出）
    - ユーザ特別扱いのハードコードを排除 — 相手の重みは関係の厚み（WM relation
      スレッド）から導出し、関係を育てれば席は大きくなり、放置すれば痩せる
    - 誤検知のコストは問い合わせ1回 — 封筒のみの粗い計算でよい
"""

import math
import random
from datetime import datetime

# 体質プロファイルの標準値（pressure_profile が NULL のキャラに適用）
DEFAULT_PROFILE: dict = {
    "version": 1,
    # tau_days: 対人接触の安らぎが薄れる時定数（日）。小さいほど早く人恋しくなる。
    # sharpness: 誰でもいい派(0.0)⇔特定の人じゃないと駄目派(1.0)。
    "social": {"tau_days": 2.5, "sharpness": 0.3},
    # sensitivity: 単調さへの感度。大きいほど退屈しやすい。
    "boredom": {"sensitivity": 1.0},
    # fatigue_sensitivity: 疲労の溜まりやすさ。大きいほど疲れを引きずる。
    "body": {"fatigue_sensitivity": 1.0},
}

# リズム成分の決定論導出に使う絶対エポック（絶対時刻に対して決定的にするため固定）
_RHYTHM_EPOCH = datetime(2026, 1, 1)
# リズム成分の固定振幅（体質インタビューでも聞かない — 誰も設計していない波）
_RHYTHM_AMPLITUDE = 0.25

# 疲労成分の減衰時定数（日）と正規化係数（この件数の直近イベントで疲労1.0相当）
_FATIGUE_TAU_DAYS = 1.5
_FATIGUE_NORM_EVENTS = 40.0
# 夢（night.chronicle）1回の回復量（イベント換算）
_CHRONICLE_RECOVERY_EVENTS = 4.0

# 退屈圧の観測窓（日）と基準値
_BOREDOM_WINDOW_DAYS = 7
_BOREDOM_BASELINE_EVENTS_PER_DAY = 12.0
_BOREDOM_DIVERSITY_NORM = 8.0

# 関係の重みが引けない相手の既定値（コールドスタート）
_DEFAULT_RELATION_WEIGHT = 0.35

# 圧力計算に読む封筒の窓（日）。社会圧の relief は tau 数日で消えるため十分な幅
_EVENT_WINDOW_DAYS = 30


def merge_profile(raw: dict | None) -> dict:
    """保存された体質プロファイルを標準値とマージして完全な形にする。

    Args:
        raw: characters.pressure_profile の値（None / 部分的な dict を許容）。

    Returns:
        DEFAULT_PROFILE の全キーが揃った dict（interview 等の付加キーは素通し）。
    """
    profile = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULT_PROFILE.items()}
    for key, value in (raw or {}).items():
        if key in ("social", "boredom", "body") and isinstance(value, dict):
            profile[key] = {**profile[key], **value}
        else:
            profile[key] = value
    return profile


def _days_between(now: datetime, then: datetime) -> float:
    """2時刻の差を日数（float）で返す。未来の時刻は 0 に丸める。"""
    return max(0.0, (now - then).total_seconds() / 86400.0)


def _partner_of(event) -> str | None:
    """封筒から「対人接触の相手」ラベルを取り出す。対人イベントでなければ None。

    - chat.message / chat.farewell / action.performed(対ユーザ) → "user"
    - scene.turn で actor が npc:<名前> → その名前（うつつのNPCとの交流も接触）
    - narrator / system / 自分の独白などは対人ではない
    """
    if event.event_type in ("chat.message", "chat.farewell"):
        return "user"
    if event.event_type == "action.performed" and event.counterpart == "user":
        return "user"
    if event.event_type == "scene.turn":
        actor = event.actor or ""
        if actor.startswith("npc:"):
            return actor[4:]
        if actor == "user":
            return "user"
    return None


def compute_social(
    events: list,
    now: datetime,
    profile: dict,
    relation_weight_fn,
) -> float:
    """社会圧 — 「人と関わっていない」の物理量を計算する。

    対人イベントの安らぎ（relief）が時間とともに指数減衰し、
    残りの安らぎが少ないほど圧が高い:

        relief = Σ_(相手,日) w_eff(相手) × exp(-経過日数 / tau_days)
        社会圧 = clamp(1 - relief, 0, 1)

    同じ相手との同日の発言は1接触に丸める（メッセージ数で安らぎが
    無限に積み上がらないように。会った・話した、が単位）。

    相手別重み: w_eff = 関係の重み ^ (1 + 3×体質の鋭さ)。
    鋭さ 0（誰でもいい派）なら軽い関係でもそのまま安らぎになるが、
    鋭さ 1（特定の人じゃないと駄目派）では軽い関係（0.35^4 ≈ 0.015）は
    ほぼ安らがず、厚い関係（0.9^4 ≈ 0.66）だけが効く。

    Args:
        events: タイムライン封筒（時系列昇順）。
        now: 基準時刻。
        profile: merge_profile 済みの体質。
        relation_weight_fn: 相手ラベル → 関係の重み(0..1) を返す関数。

    Returns:
        社会圧（0.0〜1.0）。
    """
    tau = float(profile["social"]["tau_days"])
    sharpness = float(profile["social"]["sharpness"])
    gamma = 1.0 + 3.0 * max(0.0, min(1.0, sharpness))

    # (相手, 日付) ごとに最新の接触だけを数える（同日の連続発言は1接触）
    contacts: dict[tuple, datetime] = {}
    for ev in events:
        partner = _partner_of(ev)
        if partner is None:
            continue
        key = (partner, ev.occurred_at.date())
        if key not in contacts or ev.occurred_at > contacts[key]:
            contacts[key] = ev.occurred_at

    relief = 0.0
    for (partner, _date), at in contacts.items():
        weight = max(0.0, min(1.0, float(relation_weight_fn(partner))))
        w_eff = weight ** gamma
        relief += w_eff * math.exp(-_days_between(now, at) / tau)
    return max(0.0, min(1.0, 1.0 - relief))


def compute_boredom(events: list, now: datetime, profile: dict) -> float:
    """退屈圧 — 直近タイムラインのイベント密度・多様性の低さを計算する。

    直近 _BOREDOM_WINDOW_DAYS 日の封筒について:
        密度スコア   = min(1, 1日あたり件数 / 基準値)
        多様性スコア = min(1, 種類数 / 基準値)（event_type・actor・origin の異なり数）
        退屈圧 = clamp((1 - 密度×0.5 - 多様性×0.5) × 感度, 0, 1)

    封筒のみの粗い計算でよい — 圧力は「いつ聞くか」だけを決め、
    高退屈圧→問い合わせ→「別に退屈じゃない、穏やかでいい」もまた発見。

    Args:
        events: タイムライン封筒。
        now: 基準時刻。
        profile: merge_profile 済みの体質。

    Returns:
        退屈圧（0.0〜1.0）。
    """
    sensitivity = float(profile["boredom"]["sensitivity"])
    window = [
        ev for ev in events
        if _days_between(now, ev.occurred_at) <= _BOREDOM_WINDOW_DAYS
    ]
    per_day = len(window) / _BOREDOM_WINDOW_DAYS
    density_score = min(1.0, per_day / _BOREDOM_BASELINE_EVENTS_PER_DAY)
    kinds: set = set()
    for ev in window:
        kinds.add(("type", ev.event_type))
        if ev.actor:
            kinds.add(("actor", ev.actor))
        kinds.add(("origin", ev.origin))
    diversity_score = min(1.0, len(kinds) / _BOREDOM_DIVERSITY_NORM)
    raw = 1.0 - 0.5 * density_score - 0.5 * diversity_score
    return max(0.0, min(1.0, raw * sensitivity))


def rhythm_component(character_id: str, now: datetime) -> float:
    """体調圧のリズム成分 — character_id シードから決定論導出される固有周期の波。

    周期は 7日と30日に二峰を持つ分布（対数正規2峰混合）から引き、4〜90日に
    クランプ。振幅は固定（疲労成分との合成で実効値は複雑系になる）。
    **誰も設計していない波** — 体質インタビューでも聞かない。
    乱数は世界に置く: シードが同じなら常に同じ波（絶対時刻に対して決定的）。

    Args:
        character_id: キャラクター ID（シード）。
        now: 基準時刻。

    Returns:
        リズム成分（0.0〜_RHYTHM_AMPLITUDE）。
    """
    rng = random.Random(f"meguri-rhythm:{character_id}")
    peak = 7.0 if rng.random() < 0.5 else 30.0
    period = math.exp(rng.gauss(math.log(peak), 0.35))
    period = max(4.0, min(90.0, period))
    phase = rng.random()
    days = (now - _RHYTHM_EPOCH).total_seconds() / 86400.0
    wave = math.sin(2.0 * math.pi * (days / period + phase))
    return _RHYTHM_AMPLITUDE * 0.5 * (1.0 + wave)


def compute_body(
    events: list,
    now: datetime,
    profile: dict,
    character_id: str,
) -> float:
    """体調圧 — 疲労成分（イベント密度の減衰積分）＋リズム成分（固有周期の波）。

    疲労 = Σ_(活動イベント) exp(-経過日数 / tau) / 正規化
           − Σ_(night.chronicle) 回復量 × exp(-経過日数 / tau) / 正規化

    夢中で夜更かしした翌日は疲労が溜まった状態から始まる（減衰積分なので
    「後でどっと来る」は追加実装なしに創発する）。疲労は静かな日（新規イベントが
    積まれない = 指数減衰で自然回復）と夢（night.chronicle の回復項）で回復する。

    Args:
        events: タイムライン封筒。
        now: 基準時刻。
        profile: merge_profile 済みの体質。
        character_id: リズム成分のシード。

    Returns:
        体調圧（0.0〜1.0）。
    """
    sensitivity = float(profile["body"]["fatigue_sensitivity"])
    load = 0.0
    for ev in events:
        decay = math.exp(-_days_between(now, ev.occurred_at) / _FATIGUE_TAU_DAYS)
        if ev.event_type == "night.chronicle":
            load -= _CHRONICLE_RECOVERY_EVENTS * decay
        elif ev.event_type.startswith("night."):
            continue  # 夜間バッチ自体は活動ではない
        else:
            load += decay
    fatigue = max(0.0, min(1.0, (load / _FATIGUE_NORM_EVENTS) * sensitivity))
    return max(0.0, min(1.0, fatigue + rhythm_component(character_id, now)))


def _make_relation_weight_fn(sqlite, character_id: str):
    """相手ラベル → 関係の重み(0..1) を返す関数を作る。

    関係の重み = relation 系 WM スレッド（relation_target 一致）の importance。
    見つからない相手は既定値（コールドスタート）。ユーザも特別扱いしない —
    ユーザが重いのは「キャラの記憶の中で重いから」（賭け金の実装）。

    "user" ラベルは characters.user_label（キャラがユーザを呼ぶ名前）で
    relation_target を引き直す（WM スレッドは呼称で立っているため）。
    """
    char = sqlite.get_character(character_id)
    user_label = (getattr(char, "user_label", "") or "").strip() if char else ""
    threads = sqlite.list_working_memory_threads(character_id, type="relation", is_open=True)
    weights: dict[str, float] = {}
    for t in threads:
        target = (getattr(t, "relation_target", "") or "").strip()
        if target:
            weights[target] = float(getattr(t, "importance", 0.5) or 0.5)

    def weight_fn(partner: str) -> float:
        label = user_label if (partner == "user" and user_label) else partner
        return weights.get(label, _DEFAULT_RELATION_WEIGHT)

    return weight_fn


def compute_pressures(sqlite, character_id: str, now: datetime | None = None) -> dict:
    """キャラクターの現在の圧力3変数を計算する（読み取り時計算の共通入口）。

    Args:
        sqlite: SQLiteStore（封筒と WM スレッドの読み出しに使う）。
        character_id: 対象キャラクター ID。
        now: 基準時刻。None なら現在時刻。

    Returns:
        {"social": float, "boredom": float, "body": float}（各 0.0〜1.0）。
    """
    from datetime import timedelta

    now = now or datetime.now()
    char = sqlite.get_character(character_id)
    profile = merge_profile(getattr(char, "pressure_profile", None) if char else None)
    events = sqlite.list_timeline_events(
        character_id, since=now - timedelta(days=_EVENT_WINDOW_DAYS), until=now,
    )
    weight_fn = _make_relation_weight_fn(sqlite, character_id)
    return {
        "social": compute_social(events, now, profile, weight_fn),
        "boredom": compute_boredom(events, now, profile),
        "body": compute_body(events, now, profile, character_id),
    }


def pressure_plain_lines(pressures: dict) -> list[str]:
    """圧力を「生に近い淡白な一行」へ変換する（プロンプト注入用）。

    解釈済みの言葉ではなく物理の報告に留める — どう感じるか・WM body に
    何を書くかはキャラクターに任せる（圧＝物理、WM＝意味、の分業）。
    低圧のものは何も言わない（沈黙も情報）。

    Args:
        pressures: compute_pressures の戻り値。

    Returns:
        淡白な一行のリスト（全部低圧なら空リスト）。
    """
    lines: list[str] = []
    body = pressures.get("body", 0.0)
    social = pressures.get("social", 0.0)
    boredom = pressures.get("boredom", 0.0)
    if body >= 0.8:
        lines.append("ここ数日、体はかなり重い。")
    elif body >= 0.6:
        lines.append("ここ数日、体は重め。")
    if social >= 0.8:
        lines.append("ずいぶん長いこと、人とゆっくり話していない。")
    elif social >= 0.6:
        lines.append("しばらく人とゆっくり話していない。")
    if boredom >= 0.8:
        lines.append("ここのところ、日々はずっと単調。")
    elif boredom >= 0.6:
        lines.append("ここのところ、日々は少し単調。")
    return lines


def record_pressure_meters(sqlite) -> int:
    """全キャラクターの圧力3変数を計器メーターとして日次スナップショットする。

    圧力は保存しない（純関数）が、傾向観測のためのメーター記録だけは残す
    （docs/planned/aliveness_plan.md §4.1「日次スナップショットは計器メーターとして残す」）。
    計器スケジューラ（05:00）から呼ばれる。

    Args:
        sqlite: SQLiteStore。

    Returns:
        記録したスナップショット行数。
    """
    import logging
    logger = logging.getLogger(__name__)
    recorded = 0
    for char in sqlite.list_characters():
        try:
            pressures = compute_pressures(sqlite, char.id)
        except Exception:
            logger.exception("圧力スナップショットに失敗 char=%s", char.name)
            continue
        for name, value in pressures.items():
            sqlite.record_meter(f"pressure_{name}", value, character_id=char.id)
            recorded += 1
    return recorded
