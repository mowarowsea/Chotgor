"""タイムライン投影（projector）— 封筒正本の観測者別ビュー。

全プロンプト・GM入力・バッチ入力は、この投影を通した「可視性フィルタ付きの派生ビュー」
であり、正本は timeline_events 封筒だけ（docs/aliveness_plan.md §2）。

観測者クラス（3つ）:
    - self        : キャラクター本人（1on1・うつつPC・バッチ問い合わせ。差はプロンプト予算のみ）
    - world_frame : 世界を回す側（うつつGM・シナリオGM）
    - user_ui     : ユーザの画面（チャットUI・管理UI・ログUI）

開示レベル（3値）:
    - hidden   : 存在ごと見えない
    - envelope : 封筒固定カラムのみ（中身は見えない）
    - content  : 中身まで見える

ポリシーは (observer, event_type 名前空間, origin) → 開示レベル のコード内テーブル
＋ user_ui ダイヤル修飾。判定結果は封筒行に焼き付けない（読み取り時ポリシー）。
retracted な封筒は全観測者から hidden（store 層の読み出しで既に除外される）。
"""

import json
from dataclasses import dataclass
from datetime import datetime

# 観測者クラスの正規値（Literal 相当。誤字をフェイルファストにするため集合も持つ）
OBSERVERS = ("self", "world_frame", "user_ui")

# 曜日の日本語表記（フォーマッタ用）
_WEEKDAYS_JA = "月火水木金土日"


@dataclass
class Budget:
    """投影の予算 — 可視性と直交する量の上限。

    Attributes:
        max_events: 返すイベント数の上限（新しい方を優先して残す）。None は無制限。
        max_chars: content 合計文字数の上限。超過分の content は封筒止めに落とす。
            None は無制限。
    """

    max_events: int | None = None
    max_chars: int | None = None


@dataclass
class ProjectedEvent:
    """投影されたイベント1件 — 封筒フィールド＋開示レベル＋（開示時のみ）中身。

    Attributes:
        disclosure: "envelope" | "content"（hidden は投影結果に現れない）。
        content: disclosure="content" のときの中身テキスト。封筒止めなら None。
        その他: timeline_events の封筒フィールドのコピー。
    """

    id: str
    character_id: str
    event_type: str
    occurred_at: datetime
    actor: str | None
    counterpart: str | None
    origin: str
    modality: str | None
    session_id: str | None
    intent_id: str | None
    payload: dict | None
    disclosure: str
    content: str | None


def resolve_disclosure(
    observer: str,
    event_type: str,
    origin: str,
    counterpart: str | None = None,
    user_dial: int = 0,
) -> str:
    """観測者ポリシー本体 — (observer, event_type, origin) を開示レベルへ写像する。

    docs/aliveness_plan.md §2.4 の表のコード化。設計判断:
        - GM への chat.*（real）は envelope 止め。封筒は「ユーザについて」の材料、
          中身は「ユーザが」の材料になり得るので渡さない。現実ログの引用権は
          self（キャラ本人）だけが持つ。
        - GM への intent.* は存在ごと hidden。秘密（性質6）は GM にも適用
          （GM が意図を知ると先回り演出＝自己成就が起きる）。
        - action.performed（対ユーザ）だけは GM に envelope。世界の接触事実であり、
          隠すと因果的一貫性（性質4）が破れる。
        - 未知の event_type は安全側（self 以外 hidden）に倒す。

    Args:
        observer: "self" / "world_frame" / "user_ui"。
        event_type: 封筒の event_type（ドット記法）。
        origin: real / usual / interlude。
        counterpart: 封筒の「相手」。action.performed の対ユーザ判定に使う。
        user_dial: user_ui のダイヤル段階（0=全開〜3=最終形）。他の観測者では無視。

    Returns:
        "hidden" | "envelope" | "content"。
    """
    if observer not in OBSERVERS:
        raise ValueError(f"未知の観測者クラス: {observer}")

    if observer == "self":
        # 本人は常に content（自分の身に起きたことは全部知っている）
        return "content"

    if observer == "world_frame":
        if event_type.startswith("chat."):
            return "envelope"
        if event_type.startswith("scene."):
            return "content"
        if event_type == "action.performed":
            return "envelope" if counterpart == "user" else "hidden"
        # memory.* / night.* / intent.* / 未知イベント
        return "hidden"

    # observer == "user_ui": ダイヤル修飾（段階は累積）
    if user_dial >= 3:
        # 最終形: チャット応答のみ（計器はダイヤル非依存で別系統）
        return "content" if event_type == "chat.message" else "hidden"
    if user_dial >= 2:
        # 内面の秘匿
        if (
            event_type.startswith("memory.")
            or event_type.startswith("intent.")
            or event_type.startswith("night.")
        ):
            return "hidden"
        if event_type.startswith("scene.") and origin == "usual":
            return "envelope"
        return "content"
    if user_dial >= 1:
        # 生活の秘匿
        if event_type.startswith("scene.") and origin == "usual":
            return "envelope"
        return "content"
    # ダイヤル0（全開・開発期）
    return "content"


def _payload_content(event) -> str:
    """payload 完結型イベント（source_table=None）の content 表現を組み立てる。

    型ごとに意味のあるテキストへ整形し、未知の型は payload の JSON をそのまま返す。
    """
    payload = event.payload or {}
    if event.event_type == "memory.carved":
        return str(payload.get("content") or "")
    if event.event_type == "chat.farewell":
        return str(payload.get("reason") or "")
    return json.dumps(payload, ensure_ascii=False) if payload else ""


def project(
    character_id: str,
    observer: str,
    sqlite,
    since: datetime | None = None,
    until: datetime | None = None,
    origins: list[str] | None = None,
    types: list[str] | None = None,
    budget: Budget | None = None,
    user_dial: int = 0,
) -> list[ProjectedEvent]:
    """封筒正本を観測者ポリシーで投影する（読み取り側の共通入口）。

    Args:
        character_id: 誰のタイムラインを見るか。
        observer: 観測者クラス（"self" / "world_frame" / "user_ui"）。
        sqlite: SQLiteStore（timeline_events と中身テーブルの読み出しに使う）。
        since: この時刻以降（inclusive）。
        until: この時刻以前（exclusive）。
        origins: origin フィルタ（["real"] 等）。None なら全 origin。
        types: event_type フィルタ。"chat.*" のような名前空間指定と
            "chat.message" のような完全指定の両方を受ける。
        budget: イベント数・文字数上限（可視性と直交）。新しいイベントを優先して残す。
        user_dial: user_ui のダイヤル段階（0〜3）。observer="user_ui" 以外では無視。

    Returns:
        occurred_at 昇順の ProjectedEvent リスト。hidden 判定のイベントは含まれない。
    """
    # "chat.*" → 前方一致プレフィックス "chat." に正規化
    prefixes = None
    if types:
        prefixes = [t[:-1] if t.endswith("*") else t for t in types]

    rows = sqlite.list_timeline_events(
        character_id,
        since=since,
        until=until,
        origins=origins,
        event_type_prefixes=prefixes,
    )

    # 1. 可視性フィルタ（hidden を落とし、開示レベルを付与）
    visible: list[tuple] = []  # (row, disclosure)
    for row in rows:
        disclosure = resolve_disclosure(
            observer, row.event_type, row.origin,
            counterpart=row.counterpart, user_dial=user_dial,
        )
        if disclosure == "hidden":
            continue
        visible.append((row, disclosure))

    # 2. イベント数予算（新しい方を優先して残す）
    if budget and budget.max_events is not None and len(visible) > budget.max_events:
        visible = visible[-budget.max_events:]

    # 3. content 開示分の中身を source テーブルから一括取得
    ids_by_table: dict[str, list[str]] = {}
    for row, disclosure in visible:
        if disclosure == "content" and row.source_table and row.source_id:
            ids_by_table.setdefault(row.source_table, []).append(row.source_id)
    contents: dict[tuple[str, str], str] = {}
    for table, ids in ids_by_table.items():
        fetched = sqlite.fetch_timeline_source_contents(table, ids)
        for sid, text in fetched.items():
            contents[(table, sid)] = text

    # 4. ProjectedEvent へ整形（文字数予算は新しい方から消費し、超過分は封筒止めへ格下げ）
    results: list[ProjectedEvent] = []
    remaining_chars = budget.max_chars if (budget and budget.max_chars is not None) else None
    # 文字数予算は「新しいイベント優先」なので逆順に舐めてから戻す
    for row, disclosure in reversed(visible):
        content: str | None = None
        if disclosure == "content":
            if row.source_table and row.source_id:
                content = contents.get((row.source_table, row.source_id), "")
            else:
                content = _payload_content(row)
            if remaining_chars is not None:
                if len(content) <= remaining_chars:
                    remaining_chars -= len(content)
                else:
                    # 予算切れ: 中身を落として封筒止めに格下げ（存在は残す）
                    disclosure = "envelope"
                    content = None
        results.append(ProjectedEvent(
            id=row.id,
            character_id=row.character_id,
            event_type=row.event_type,
            occurred_at=row.occurred_at,
            actor=row.actor,
            counterpart=row.counterpart,
            origin=row.origin,
            modality=row.modality,
            session_id=row.session_id,
            intent_id=row.intent_id,
            payload=row.payload,
            disclosure=disclosure,
            content=content,
        ))
    results.reverse()
    return results


def _format_dt(dt: datetime) -> str:
    """封筒の時刻を「MM/DD(曜) HH:MM」形式の短い日本語表記にする。"""
    return f"{dt.month}/{dt.day}({_WEEKDAYS_JA[dt.weekday()]}) {dt:%H:%M}"


def format_real_contact_block(
    events: list[ProjectedEvent],
    character_name: str,
    user_label: str,
) -> str:
    """うつつ GM 向け「現実の接触の記録」ブロックを封筒（envelope）から組み立てる。

    性質4（因果的一貫性）の穴埋め: GM が「キャラとユーザがいつ・どのくらい接触したか」の
    外形を知ることで、NPC の呼び水（「昨日あの人と話したんだって？」）が現実と矛盾しなく
    なる。**中身（何を話したか）は絶対に載せない** — 中身を場に持ち込めるのはキャラ本人
    だけ（主語ベース3段ルールと同じ思想。docs/aliveness_plan.md §2.4）。

    連続する chat.message 封筒はセッション×日付単位で「HH:MM〜HH:MM ごろ N 往復」に
    集約する（原子イベントで記録し、集約は投影が導出する — §2.3 (b)）。

    Args:
        events: observer="world_frame" で投影済みのイベント列
            （chat.message / chat.farewell / action.performed を想定。他は無視）。
        character_name: キャラクター名（表示用）。
        user_label: ユーザの呼称。空ならブロックを生成しない。

    Returns:
        GM へ渡す OOC ブロック文字列。素材ゼロまたは user_label 空なら空文字列。
    """
    label = (user_label or "").strip()
    name = (character_name or "").strip() or "本人"
    if not label:
        return ""

    # chat.message をセッション×日付で集約し、farewell / action は単発行にする
    lines: list[str] = []
    chat_groups: dict[tuple, dict] = {}  # (session_id, date) -> {first, last, count, modality}
    singles: list[tuple[datetime, str]] = []
    for ev in events:
        if ev.event_type == "chat.message":
            key = (ev.session_id, ev.occurred_at.date())
            g = chat_groups.setdefault(key, {
                "first": ev.occurred_at, "last": ev.occurred_at,
                "count": 0, "face": False,
            })
            g["first"] = min(g["first"], ev.occurred_at)
            g["last"] = max(g["last"], ev.occurred_at)
            g["count"] += 1
            g["face"] = g["face"] or (ev.modality == "face")
        elif ev.event_type == "chat.farewell":
            singles.append((
                ev.occurred_at,
                f"{_format_dt(ev.occurred_at)} {name} は自分から会話を切り上げた",
            ))
        elif ev.event_type == "action.performed":
            singles.append((
                ev.occurred_at,
                f"{_format_dt(ev.occurred_at)} {name} から {label} へ働きかけがあった（連絡など）",
            ))

    entries: list[tuple[datetime, str]] = list(singles)
    for (_sid, _date), g in chat_groups.items():
        mode = "対面で" if g["face"] else "テキストで"
        if g["first"] == g["last"]:
            span = _format_dt(g["first"])
        else:
            span = f"{_format_dt(g['first'])}〜{g['last']:%H:%M}"
        entries.append((
            g["first"],
            f"{span} ごろ、{label} と{mode}やり取りした（{g['count']}発言分）",
        ))
    if not entries:
        return ""
    entries.sort(key=lambda e: e[0])
    lines = [
        f"# {label} との現実の接触の記録（外形のみ・内容は不明）",
        f"@{name} と {label} の現実のやり取りの「あった・なかった」だけをあなたは知っている。"
        f"**何を話したかは知らない**（中身に触れられるのは @{name} 本人だけ）。"
        f"NPC の伝聞・呼び水の整合性にだけ使うこと:",
        "",
    ]
    lines += [f"- {text}" for _, text in entries]
    return "\n".join(lines)
