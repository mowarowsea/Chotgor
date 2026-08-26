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

# 疲労成分の減衰時定数（日）。τ=1.5 は「1日で約半分（48.7%）回復」に相当し、
# 静かな日の自然減衰そのものが回復として働く（睡眠を別イベントとして数えない —
# 夜間バッチの成否に依存しない堅い回復）。
_FATIGUE_TAU_DAYS = 1.5
# 定常状態の load を平常件数/日 から導く係数。
# 封筒は日単位に固まって並ぶ離散データなので、連続近似の τ ではなく等比級数の和
# Σ_d exp(-d/τ) = 1/(1-exp(-1/τ)) を使う（τ=1.5 で 2.055）。τ で代用すると当日分が
# 減衰なしで満額入るぶん平常 load を 1.37 倍過小に見積もり、平常運転の体調圧が
# 中央(0.5)ではなく 0.64 に浮く。
_FATIGUE_STEADY_FACTOR = 1.0 / (1.0 - math.exp(-1.0 / _FATIGUE_TAU_DAYS))
# 疲労は固定の魔法定数ではなく、キャラ自身の平常活動量との**比**で測る:
#   疲労 = 0.5 + 0.5 × log(load / 平常load) / log(_FATIGUE_HEADROOM)
# 「平常運転＝0.5、平常の HEADROOM 倍＝1.0、平常の 1/HEADROOM 倍＝0.0」。平常は
# キャラのタイムラインから導出するので、よく喋る子は基準が上がり同じ量では疲れない
# （封筒から導く思想）。対数なので値がそのまま「平常の何倍か」を表し、疲労離席の
# θ_hard を絶対値のまま意味づけできる。
_FATIGUE_HEADROOM = 3.0
# 平常件数/日 を測る観測窓（日）。短めにして直近の生活リズムへ追従させる
# （週末たくさん→週明けに疲れ残り、といった週リズムが出る塩梅）。
_FATIGUE_BASELINE_WINDOW_DAYS = 30
# 平常が信頼して測れる最小データ日数。これ未満はコールドスタート扱い。
_FATIGUE_BASELINE_MIN_DAYS = 7
# コールドスタート時に仮定する平常件数/日。既定の平常 load ≈ 45×2.055 ≈ 92 となり、
# データが貯まるまでは「1日90件クラスの活動が平常」の固定基準として振る舞う。
_FATIGUE_DEFAULT_RATE = 45.0
# 平常件数/日 の下限（寡動キャラでも極端に小さい平常 load にしない安全弁）。
_FATIGUE_MIN_RATE = 15.0
# load の下限。完全無活動（load=0）で log(0) が発散するのを防ぐ。
# この値なら ratio が下限クランプ（疲労0.0）に十分届く。
_FATIGUE_MIN_RATIO = 1e-6

# 退屈圧の観測窓（日）。直近窓と「それ以前」を比べて新規性を測る。
_BOREDOM_WINDOW_DAYS = 7
# 新規性の比較対象にする「それ以前」の窓（日）。7+23=30 で封筒の取得窓と一致する。
_BOREDOM_LOOKBACK_DAYS = 23
# 密度と新規性の配合。w = _BOREDOM_W_BASE + _BOREDOM_W_SPAN × (1 - 体調圧) が
# 新規性側の重みになる（体調が良い日ほど「代わり映えのなさ」が効く）。
_BOREDOM_W_BASE = 0.3
_BOREDOM_W_SPAN = 0.4

# 関係の重みが引けない相手の既定値（コールドスタート）
_DEFAULT_RELATION_WEIGHT = 0.35

# --- 発話閾値の分位点化（docs/planned/aliveness_plan.md §4.1「表現」）---
# 絶対値の閾値は「その定式化のときたまたま噛み合っていた数字」でしかなく、物理量の
# 定式化を触るたびに言いすぎ／言わなすぎへ倒れる。閾値はキャラ自身の分布から取る。
_SPEECH_WINDOW_DAYS = 60          # 分位点を引くメーター履歴の窓
# 圧力の定式化を変えた日。これより前のメーターは**別の物差しで測った値**なので
# 分位点の材料にしない（例: 再定式化で社会圧は 0.63 → 0.00、退屈圧は 0.00 → 0.51 と
# 分布ごと動いた。混ぜると新しい値が一律「下位」「上位」に貼りつく）。
# 履歴を消す代わりに読む範囲を切る — 圧力は純関数なので、計器メーターは観測用として
# そのまま残しておいてよい。**今後また定式化を変えたらこの日付を更新すること。**
_FORMULA_EPOCH = datetime(2026, 8, 26)
_SPEECH_MIN_SAMPLES = 20          # これ未満は絶対値へフォールバック（日次1点なので約3週間）
_SPEECH_MIN_IQR = 0.05            # 分布がこれより平坦な圧は沈黙させる（下記参照）
_SPEECH_Q_HIGH = 0.90             # 強い表現（HIGH プール）
_SPEECH_Q_MID = 0.75              # 穏やかな表現（MID プール）
_SPEECH_Q_GOOD = 0.20             # 好調（GOOD プール）
# ウォームアップ中・分位点が引けないときの絶対値（high, mid, good）
_SPEECH_ABS_THRESHOLDS = (0.8, 0.6, 0.2)

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


def _partner_of(event, self_name: str | None = None) -> str | None:
    """封筒から「対人接触の相手」ラベルを取り出す。対人イベントでなければ None。

    - chat.message / chat.farewell / action.performed(対ユーザ) → "user"
    - scene.turn で actor が npc:<名前> → その名前（うつつのNPCとの交流も接触）
    - narrator / system / 自分の独白などは対人ではない

    Args:
        event: 封筒。
        self_name: キャラクター本人の名前。うつつの GM は PC 本人の発話も
            `@<本人名>:` で書くため、渡さないと**自分との会話で社会圧が下がる**
            （はるの実測で30日に89件混入していた）。
    """
    if event.event_type in ("chat.message", "chat.farewell"):
        return "user"
    if event.event_type == "action.performed" and event.counterpart == "user":
        return "user"
    if event.event_type == "scene.turn":
        actor = event.actor or ""
        if actor.startswith("npc:"):
            name = actor[4:]
            if self_name and name == self_name:
                return None  # 自分は対人接触の相手ではない
            return name
        if actor == "user":
            return "user"
    return None


def _canonical_label_map(labels: set[str]) -> dict[str, str]:
    """表記揺れした相手ラベルを、より詳しい表記へ寄せる対応表を作る。

    うつつの NPC 名は GM が `@名前:` に書いたものがそのまま封筒へ入り、正規名の
    辞書は存在しない（未知 NPC をそのまま通す設計）。このため同一人物が
    「ひろこ」「田中ひろこ」のように分裂し、relation スレッドと突合できずに
    コールドスタート既定値へ落ちる（＝関係を育てても社会圧が下がらない）。
    同一視は封筒に現れたラベル集合だけから導く:

        短いラベルが長いラベルの部分文字列で、その長いラベルが**一意**なら寄せる。

    候補が複数あるとき（「佐藤」に対し「佐藤彰」と「佐藤花子」がある）は誰なのか
    決められないため寄せない — 取り違えるくらいは分裂したままにしておく。
    "user" は表記揺れしないので対象外。

    Args:
        labels: 封筒から集めた相手ラベルの集合。

    Returns:
        {揺れたラベル: 正規ラベル}。寄せ先がないものは含まない。
    """
    names = sorted(l for l in labels if l != "user")
    mapping: dict[str, str] = {}
    for short in names:
        longer = [l for l in names if l != short and short in l]
        if len(longer) == 1:
            mapping[short] = longer[0]
    return mapping


# 同日の接触が「濃い」ほど安らぎを底上げする頭打ち付きブースト。
# 件数そのものを無制限に効かせるとメッセージスパムで安らぎが積み上がってしまうため、
# _SOCIAL_RICHNESS_CAP_COUNT 件目以降は伸びを止める（会話量は反映するが青天井にはしない）。
_SOCIAL_RICHNESS_GAIN = 0.15
_SOCIAL_RICHNESS_CAP_COUNT = 6
# 対面接触はテキストより厚く安らぐ（同じ回数でも質が違う）。
_SOCIAL_FACE_MULT = 1.4


def compute_social(
    events: list,
    now: datetime,
    profile: dict,
    relation_weight_fn,
    self_name: str | None = None,
    top_weight: float | None = None,
) -> float:
    """社会圧 — 「人と関わっていない」の物理量を計算する。

    対人イベントの安らぎ（relief）が時間とともに指数減衰し、
    残りの安らぎが少ないほど圧が高い:

        relief = Σ_(相手,日) w_eff(相手) × richness(相手,日) × face_mult(相手,日)
                   × exp(-経過日数 / tau_days)
        社会圧 = clamp(1 - relief, 0, 1)

    同じ相手との同日の接触は日単位で集約するが、二値（会った/会っていない）には
    しない。件数が多い日ほど richness が頭打ち付きで安らぎを底上げし
    （_SOCIAL_RICHNESS_CAP_COUNT 件で頭打ち）、対面接触があった日は
    _SOCIAL_FACE_MULT 倍テキストより厚く安らぐ。一言挨拶の日と、対面で
    しっかり話した日を同じ扱いにしないための調整（メッセージ数を無制限に
    効かせるわけではない）。

    相手別重み: w_eff = 関係の重み^gamma ÷ (そのキャラの最も厚い関係の重み^gamma)、
    gamma = 1 + 3×体質の鋭さ。鋭さ 0（誰でもいい派）なら軽い関係でもそのまま
    安らぎになるが、鋭さ 1（特定の人じゃないと駄目派）では本命との差が開き、
    軽い関係ではほとんど安らがなくなる。

    **最も厚い関係で正規化する**のは、鋭さに「どれだけ安らぐか」の総量まで
    削らせないため。冪乗の生値を使うと、鋭い体質のキャラは本命と会っていても
    安らげない（実測: sharpness=0.9・本命の重み0.45 で 0.45^3.7 = 0.052 まで潰れ、
    毎日会話していても社会圧が 0.6 台に張り付いた）。冪乗の生値は「関係の重みが
    0.9 前後ある」ことを暗黙の前提にしていたが、WM スレッドの importance は
    そのスケールで運用されていない。鋭さが決めるのは**誰と安らぐかの選択性**だけにする。

    Args:
        events: タイムライン封筒（時系列昇順）。
        now: 基準時刻。
        profile: merge_profile 済みの体質。
        relation_weight_fn: 相手ラベル → 関係の重み(0..1) を返す関数。
        self_name: キャラクター本人の名前（自分との会話を接触から除くために使う）。
        top_weight: そのキャラが持つ**関係全体**の中で最も厚い重み（正規化の分母）。
            None なら「今回接触した相手の中の最厚」で代用する。代用は縮退であって
            等価ではない — 本命と会えていない週に薄い相手とだけ会うと、その相手が
            分母になって満額安らいでしまうため、実運用では必ず渡すこと。

    Returns:
        社会圧（0.0〜1.0）。
    """
    tau = float(profile["social"]["tau_days"])
    sharpness = float(profile["social"]["sharpness"])
    gamma = 1.0 + 3.0 * max(0.0, min(1.0, sharpness))

    # 表記揺れを寄せるため、先に封筒へ現れる相手ラベルを集める
    canon = _canonical_label_map({
        p for p in (_partner_of(ev, self_name) for ev in events) if p is not None
    })

    # (相手, 日付) ごとに集約: 最新時刻・件数・対面接触の有無
    contacts: dict[tuple, dict] = {}
    for ev in events:
        partner = _partner_of(ev, self_name)
        if partner is None:
            continue
        partner = canon.get(partner, partner)
        key = (partner, ev.occurred_at.date())
        bucket = contacts.setdefault(key, {"at": ev.occurred_at, "count": 0, "face": False})
        bucket["count"] += 1
        if ev.occurred_at > bucket["at"]:
            bucket["at"] = ev.occurred_at
        if getattr(ev, "modality", None) == "face":
            bucket["face"] = True

    # 最も厚い関係を分母に置く（鋭さは選択性だけを決め、総量は削らない）
    raw_weights = {
        partner: max(0.0, min(1.0, float(relation_weight_fn(partner)))) ** gamma
        for partner, _date in contacts
    }
    if top_weight is not None:
        top = max(0.0, min(1.0, float(top_weight))) ** gamma
    else:
        top = max(raw_weights.values(), default=0.0)

    relief = 0.0
    for (partner, _date), info in contacts.items():
        w_eff = min(1.0, raw_weights[partner] / top) if top > 0 else 0.0
        richness = 1.0 + _SOCIAL_RICHNESS_GAIN * (
            min(info["count"], _SOCIAL_RICHNESS_CAP_COUNT) - 1
        )
        face_mult = _SOCIAL_FACE_MULT if info["face"] else 1.0
        relief += w_eff * richness * face_mult * math.exp(-_days_between(now, info["at"]) / tau)
    return max(0.0, min(1.0, 1.0 - relief))


def _event_kinds(events: list, canon: dict) -> set:
    """封筒の「種類」集合を作る（event_type・actor・origin の異なり）。

    actor の NPC 名は表記揺れを寄せてから数える。寄せないと「ひろこ」と
    「田中ひろこ」が別種になり、**同じ相手と会っているのに新顔が現れたように
    見える**（新規性を過大評価する）。
    """
    kinds: set = set()
    for ev in events:
        kinds.add(("type", ev.event_type))
        actor = ev.actor
        if actor:
            if actor.startswith("npc:"):
                name = actor[4:]
                actor = "npc:" + canon.get(name, name)
            kinds.add(("actor", actor))
        kinds.add(("origin", ev.origin))
    return kinds


def compute_boredom(
    events: list, now: datetime, profile: dict, body: float | None = None,
) -> float:
    """退屈圧 — 生活の単調さ。密度の低さと新規性の乏しさから計算する。

        密度   = min(1, 直近窓の件数/日 ÷ 平常件数/日)
        新規性 = |直近窓の種類 − それ以前の種類| ÷ |直近窓の種類|
        w      = _BOREDOM_W_BASE + _BOREDOM_W_SPAN × (1 - 体調圧)
        退屈圧 = clamp((1 - (1-w)×密度 - w×新規性) × 感度, 0, 1)

    2成分ともキャラ自身の平常からの**相対**で測る（体調圧と同じ平常を共有する）。
    かつては密度基準12件/日・多様性基準8種の固定値だったが、実測（43.9件/日・27種）に対して
    3.7倍・3.4倍で両方飽和し退屈圧が 0.00 に固定される一方、イベントの少ないキャラは
    1.00 に固定され、**実質2値**になっていた。

    「多様性＝異なり数の絶対値」をやめて新規性にしたのは、**反復を検出できない**ため。
    毎日同じ顔ぶれ・同じ種類が繰り返されても異なり数は満点になる。実測では直近7日の
    27種のうち新顔は2種（7%、しかも表記揺れ由来で実質ゼロ）で、実際には単調な生活を
    「多様性満点」と誤判定していた。退屈の本質は密度ではなく新規性にある。

    **配合 w を体調圧で変調する**のは、疲れている日に刺激の乏しさが効かないようにするため
    （疲労時に新奇性希求が落ちる生理と一致）。圧力どうしを結合させる唯一の箇所で、
    3圧独立の原則に対する意図的な例外。表現としても「しんどい＋沈黙」「好調＋単調」の
    2つの一行が同じ方向を指し、ユーザ側に「今は放っておこう／今は話しかけよう」が伝わる。

    封筒のみの粗い計算でよい — 圧力は「いつ聞くか」だけを決め、
    高退屈圧→問い合わせ→「別に退屈じゃない、穏やかでいい」もまた発見。

    Args:
        events: タイムライン封筒。
        now: 基準時刻。
        profile: merge_profile 済みの体質。
        body: 体調圧（配合の変調に使う）。None なら 0.5 で縮退する。

    Returns:
        退屈圧（0.0〜1.0）。
    """
    sensitivity = float(profile["boredom"]["sensitivity"])
    recent = [
        ev for ev in events
        if _days_between(now, ev.occurred_at) <= _BOREDOM_WINDOW_DAYS
    ]
    older = [
        ev for ev in events
        if _BOREDOM_WINDOW_DAYS < _days_between(now, ev.occurred_at)
        <= _BOREDOM_WINDOW_DAYS + _BOREDOM_LOOKBACK_DAYS
    ]

    per_day = len(recent) / _BOREDOM_WINDOW_DAYS
    baseline = _baseline_activity_rate(events, now)
    density = min(1.0, per_day / baseline) if baseline > 0 else 0.0

    canon = _canonical_label_map({
        ev.actor[4:] for ev in events if ev.actor and ev.actor.startswith("npc:")
    })
    kinds_recent = _event_kinds(recent, canon)
    kinds_older = _event_kinds(older, canon)
    novelty = (
        len(kinds_recent - kinds_older) / len(kinds_recent) if kinds_recent else 0.0
    )

    w = _BOREDOM_W_BASE + _BOREDOM_W_SPAN * (1.0 - (0.5 if body is None else body))
    raw = 1.0 - (1.0 - w) * density - w * novelty
    return max(0.0, min(1.0, raw * sensitivity))


def rhythm_component(character_id: str, now: datetime) -> float:
    """体調圧のリズム成分 — character_id シードから決定論導出される固有周期の波。

    周期は 7日と30日に二峰を持つ分布（対数正規2峰混合）から引き、4〜90日に
    クランプ。振幅は固定（疲労成分との合成で実効値は複雑系になる）。
    **誰も設計していない波** — 体質インタビューでも聞かない。
    乱数は世界に置く: シードが同じなら常に同じ波（絶対時刻に対して決定的）。

    **中心はゼロ**（±振幅/2 で振れる）。波が好調側へ振れなければ「リズム」ではない —
    以前は `0.5 × (1 + wave)` の片側加算で 0〜+振幅しか動かず、平均 +振幅/2 の
    恒常的な下駄になっていた（谷でも「下駄が消える」だけで好調な日を作れなかった）。

    Args:
        character_id: キャラクター ID（シード）。
        now: 基準時刻。

    Returns:
        リズム成分（-_RHYTHM_AMPLITUDE/2 〜 +_RHYTHM_AMPLITUDE/2）。
    """
    rng = random.Random(f"meguri-rhythm:{character_id}")
    peak = 7.0 if rng.random() < 0.5 else 30.0
    period = math.exp(rng.gauss(math.log(peak), 0.35))
    period = max(4.0, min(90.0, period))
    phase = rng.random()
    days = (now - _RHYTHM_EPOCH).total_seconds() / 86400.0
    wave = math.sin(2.0 * math.pi * (days / period + phase))
    return _RHYTHM_AMPLITUDE * 0.5 * wave


def _baseline_activity_rate(events: list, now: datetime) -> float:
    """キャラの「平常の活動量（件/日）」を封筒から導く（疲労の動的正規化に使う）。

    当日を除く観測窓内の活動イベント（night.* は活動ではないので除外）を、最古の
    活動日から昨日までの実データ日数で割った平均。ゼロ件の静かな日も分母に含める
    （calendar-day 除算）ため、週末だけ喋るような偏った生活でも平常が跳ね上がらない。

    データが乏しい間（_FATIGUE_BASELINE_MIN_DAYS 未満）はコールドスタート既定値を
    返す。寡動キャラで NORM が極端に小さくならないよう下限でクランプする。

    Args:
        events: タイムライン封筒（観測窓内、時系列昇順）。
        now: 基準時刻。

    Returns:
        平常件数/日（>= _FATIGUE_MIN_RATE）。
    """
    today = now.date()
    dates = [
        ev.occurred_at.date()
        for ev in events
        if not ev.event_type.startswith("night.")
        and ev.occurred_at.date() < today
        and _days_between(now, ev.occurred_at) <= _FATIGUE_BASELINE_WINDOW_DAYS
    ]
    if not dates:
        return _FATIGUE_DEFAULT_RATE
    span_days = (today - min(dates)).days  # 最古の活動日から昨日（＝今日の前日）まで
    if span_days < _FATIGUE_BASELINE_MIN_DAYS:
        return _FATIGUE_DEFAULT_RATE
    return max(_FATIGUE_MIN_RATE, len(dates) / span_days)


def compute_body(
    events: list,
    now: datetime,
    profile: dict,
    character_id: str,
) -> float:
    """体調圧 — 疲労成分（イベント密度の減衰積分）＋リズム成分（固有周期の波）。

    load     = Σ_(活動イベント) exp(-経過日数 / tau)
    平常load = 平常件数/日 × _FATIGUE_STEADY_FACTOR   # 定常状態の load
    ratio    = load / 平常load            # 平常運転で 1.0
    疲労 = 0.5 + 0.5 × log(ratio) / log(_FATIGUE_HEADROOM)

    キャラ自身の平常活動量からの**比**で測る（固定の魔法定数を置かない）。
    **平常運転＝0.5、平常の HEADROOM 倍＝1.0、平常の 1/HEADROOM 倍＝0.0**。
    よく喋る子は基準が上がり同じ量では疲れない。夢中で夜更かしした翌日は疲労が
    溜まった状態から始まる（減衰積分なので「後でどっと来る」は追加実装なしに創発する）。
    回復は静かな日の指数減衰が担う（τ=1.5 = 1日で約半分回復）。
    夜間バッチ（night.*）は活動でも回復項でもなく無視する。

    中心を 0.5 に置くのは、**好調側にも解像度を残す**ため。かつては
    `load / (HEADROOM × 平常rate × τ)` としていたが、定常状態では load ≈ 平常rate × τ
    なので**定義上、平常運転で 1/HEADROOM ≒ 0.33 に張り付いた**（「平常＝すでに3割疲れて
    いる」）。逆に超過分だけを取る `max(0, load-平常load)/…` では平常以下が全て 0 に潰れ、
    今度は好調側が見えなくなる。対数なら平常を挟んで両側に開く。
    副産物として**値がそのまま「平常の何倍か」を表す**ので、疲労離席の θ_hard を
    絶対値のまま意味づけできる（0.95 ≒ 平常の2.0倍。§5.2）。

    体質係数 `fatigue_sensitivity` は**超過側（ratio > 1）にのみ**掛ける。
    「疲れやすさ」であって「回復しにくさ」ではないため、静かな日の落ち方までは変えない。

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
        if ev.event_type.startswith("night."):
            continue  # 夜間バッチは活動でも回復でもない（回復は減衰が担う）
        load += math.exp(-_days_between(now, ev.occurred_at) / _FATIGUE_TAU_DAYS)
    base_load = _baseline_activity_rate(events, now) * _FATIGUE_STEADY_FACTOR
    ratio = max(load, _FATIGUE_MIN_RATIO) / base_load
    fatigue = 0.5 + 0.5 * math.log(ratio) / math.log(_FATIGUE_HEADROOM)
    if ratio > 1.0:
        fatigue = 0.5 + (fatigue - 0.5) * sensitivity
    fatigue = max(0.0, min(1.0, fatigue))
    return max(0.0, min(1.0, fatigue + rhythm_component(character_id, now)))


def _make_relation_weight_fn(sqlite, character_id: str):
    """相手ラベル → 関係の重み(0..1) を返す関数と、関係全体の最厚の重みを作る。

    Returns:
        (weight_fn, top_weight)。top_weight は compute_social の正規化の分母
        （その子にとって最も厚い関係）。relation スレッドが1件も無ければ既定値。

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

    top_weight = max(weights.values(), default=_DEFAULT_RELATION_WEIGHT)
    return weight_fn, top_weight


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
    weight_fn, top_weight = _make_relation_weight_fn(sqlite, character_id)
    # 退屈圧は体調圧で配合が変わる（§4.1）ため、体調圧を先に確定させる
    body = compute_body(events, now, profile, character_id)
    return {
        "social": compute_social(
            events, now, profile, weight_fn,
            self_name=(getattr(char, "name", None) if char else None),
            top_weight=top_weight,
        ),
        "boredom": compute_boredom(events, now, profile, body=body),
        "body": body,
    }


# 淡白な一行の表現プール。同じ文言を毎回注入すると「キャラクターの口癖」として
# 固着してしまうため、閾値ごとに複数バリエーションを持たせ random.choice で揺らす
# （物理の報告であって定型句ではない、という原則を保つための揺らぎ）。
# 語彙も「重い」に寄せすぎず、しんどい／調子が出ない／肩がこる／ぐったり等ばらす。
_BODY_HIGH_LINES = (
    "ここ数日、体がかなりしんどい。",
    "この数日、ずっとぐったりしている。",
    "ここのところ、疲れがまったく抜けない。",
    "数日来、体が重くてつらい。",
    "ここ数日、肩も体も強くこっている。",
    "このところ、体調がずっと上向かない。",
)
_BODY_MID_LINES = (
    "ここ数日、体は重め。",
    "この数日、なんとなく調子が出ない。",
    "ここのところ、肩がこりやすい。",
    "数日、疲れが軽く残っている感じ。",
    "ここのところ、体が少し怠い。",
    "この数日、本調子とは言えない。",
)
_BODY_GOOD_LINES = (
    "ここ数日、体は軽くて調子がいい。",
    "この数日、疲れがすっかり抜けている。",
    "ここのところ、体の調子はすこぶる良い。",
    "数日、体が軽やかでよく動く。",
)
_SOCIAL_HIGH_LINES = (
    "ずいぶん長いこと、人とゆっくり話していない。",
    "かなり長い間、誰かと落ち着いて話せていない。",
    "だいぶ長く、人と過ごす時間から遠ざかっている。",
    "しばらくどころではなく、人恋しさが募っている。",
    "ここのところ、誰かと言葉を交わす機会がずっと少ない。",
    "だいぶ、人との距離を感じる日々が続いている。",
)
_SOCIAL_MID_LINES = (
    "しばらく人とゆっくり話していない。",
    "ここのところ、人と話す時間が少し空いている。",
    "少し前から、誰かと話す機会が減っている。",
    "ここ最近、会話の間隔が少し空きがち。",
    "このところ、人と過ごす時間がやや少なめ。",
    "少し、人恋しさを感じる頃合い。",
)
_SOCIAL_GOOD_LINES = (
    "最近、人と過ごす時間がちゃんと満ちている。",
    "ここのところ、誰かと話す機会に恵まれている。",
    "このところ、人との距離がちょうどいい。",
    "最近、人とのやり取りに満たされている感じ。",
)
_BOREDOM_HIGH_LINES = (
    "ここのところ、日々はずっと単調。",
    "ここ最近、毎日が代わり映えしない。",
    "しばらく、同じことの繰り返しに飽き飽きしている。",
    "このところ、日々に刺激が乏しい。",
    "ここのところ、退屈がずっと続いている。",
    "ここ最近、変化のない毎日が続いている。",
)
_BOREDOM_MID_LINES = (
    "ここのところ、日々は少し単調。",
    "ここ最近、毎日がやや代わり映えしない。",
    "少し前から、同じような日々が続いている。",
    "このところ、日々に少し刺激が欲しい感じ。",
    "ここのところ、ちょっと退屈気味。",
    "この頃、変化に乏しい日々。",
)
_BOREDOM_GOOD_LINES = (
    "ここのところ、日々にちょうどいい変化がある。",
    "最近、退屈とは無縁の日々。",
    "このところ、毎日に程よい刺激がある。",
    "最近、飽きることなく過ごせている。",
)


def _quantile(sorted_values: list[float], q: float) -> float:
    """ソート済み数列の分位点を線形補間で返す（外部依存を増やさないための小実装）。"""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def compute_speech_thresholds(sqlite, character_id: str, now: datetime | None = None) -> dict:
    """圧ごとの発話閾値を、キャラ自身のメーター履歴の分位点から決める。

    絶対値の閾値は「その定式化のときたまたま噛み合っていた数字」でしかなく、
    物理量の定式化を変えるたびに言いすぎ／言わなすぎへ倒れる（実際に、旧・疲労成分では
    平常運転が 0.46 に張り付き、好調ライン 0.2 へ構造的に到達できなかった）。
    「その子にとって重い日／軽い日」を分布から決めれば、定式化がラフでも表現は偏らない。

    **平坦ガード**: 分位点は分布がどれだけ平坦でも必ず上位10%を作るため、素のままだと
    毎日どれかの圧が何かを言う状態になる（60日ずっと 0.44〜0.46 でも最上位の日は喋る）。
    分布の幅（IQR）が `_SPEECH_MIN_IQR` 未満の圧は**絶対値閾値へ戻す**。判定は圧ごと —
    体調は平坦でも社会圧は動いている、という状態があるため。

    「平坦なら黙る」ではなく「平坦なら絶対値」なのは、張り付き先が中庸とは限らないから。
    中庸に平坦なら絶対値でもどの帯にも入らず結局沈黙する（目的は達せられる）が、端に
    張り付いている圧（毎日会えていて社会圧がゼロ、など）は**その事実を言えたほうがよい**。
    黙らせるとこの情報まで落ちる。

    **移行期**: `_FORMULA_EPOCH` より前のメーターは読まない（別の物差しの値のため）。
    再定式化の直後は必然的にウォームアップ扱いになり、絶対値閾値で動く。

    Args:
        sqlite: SQLiteStore（メーター履歴の読み出しに使う）。
        character_id: 対象キャラクター ID。
        now: 基準時刻。None なら現在時刻。

    Returns:
        {"body"/"social"/"boredom": (high, mid, good)}。
        キーが無い圧は絶対値閾値へフォールバックする（ウォームアップ中・分布が平坦・
        メーター読み出しに失敗したとき）。
    """
    from datetime import timedelta

    now = now or datetime.now()
    since = max(now - timedelta(days=_SPEECH_WINDOW_DAYS), _FORMULA_EPOCH)
    thresholds: dict = {}
    for name in ("body", "social", "boredom"):
        try:
            rows = sqlite.list_meter_snapshots(
                meter_id=f"pressure_{name}", character_id=character_id, since=since,
            )
        except Exception:
            continue  # 読めなければ絶対値フォールバック（キーを置かない）
        values = sorted(float(r.value) for r in rows)
        if len(values) < _SPEECH_MIN_SAMPLES:
            continue  # ウォームアップ中
        iqr = _quantile(values, 0.75) - _quantile(values, 0.25)
        if iqr < _SPEECH_MIN_IQR:
            continue  # 平坦すぎる — 分位点に意味がないので絶対値へ戻す
        thresholds[name] = (
            _quantile(values, _SPEECH_Q_HIGH),
            _quantile(values, _SPEECH_Q_MID),
            _quantile(values, _SPEECH_Q_GOOD),
        )
    return thresholds


def _pick_line(value: float, thresholds, high_pool, mid_pool, good_pool) -> str | None:
    """1つの圧について、閾値に照らして淡白な一行を選ぶ（該当なしなら None＝沈黙）。"""
    high, mid, good = thresholds
    if value >= high:
        return random.choice(high_pool)
    if value >= mid:
        return random.choice(mid_pool)
    if value <= good:
        return random.choice(good_pool)
    return None


def pressure_plain_lines(pressures: dict, thresholds: dict | None = None) -> list[str]:
    """圧力を「生に近い淡白な一行」へ変換する（プロンプト注入用）。

    解釈済みの言葉ではなく物理の報告に留める — どう感じるか・WM body に
    何を書くかはキャラクターに任せる（圧＝物理、WM＝意味、の分業）。
    閾値ごとに表現プールを持ち random.choice で選ぶことで、同じ文言が
    固定の口癖になるのを避ける。高圧・中圧だけでなく、きわめて低圧
    （好調）のときも一行報告する。中間域（ニュートラル）だけは
    何も言わない（沈黙も情報）。

    Args:
        pressures: compute_pressures の戻り値。
        thresholds: compute_speech_thresholds の戻り値。省略・キー欠落時はその圧に
            絶対値閾値を使う（ウォームアップ中・分布が平坦なときの縮退）。

    Returns:
        淡白な一行のリスト（全部ニュートラルなら空リスト）。
    """
    thresholds = thresholds or {}
    pools = {
        "body": (_BODY_HIGH_LINES, _BODY_MID_LINES, _BODY_GOOD_LINES),
        "social": (_SOCIAL_HIGH_LINES, _SOCIAL_MID_LINES, _SOCIAL_GOOD_LINES),
        "boredom": (_BOREDOM_HIGH_LINES, _BOREDOM_MID_LINES, _BOREDOM_GOOD_LINES),
    }
    lines: list[str] = []
    for name in ("body", "social", "boredom"):
        th = thresholds.get(name, _SPEECH_ABS_THRESHOLDS)
        line = _pick_line(pressures.get(name, 0.0), th, *pools[name])
        if line:
            lines.append(line)
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
