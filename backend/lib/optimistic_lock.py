"""設定フォームの楽観ロック — フォームが書き込む値の指紋で「先祖返り」を防ぐ。

スマホと PC で同じ編集画面を開いていると、古い値を表示したままのフォームが
（自動保存で）全フィールドを丸ごと送り、他端末の変更を巻き戻してしまう。
これを検出するために、そのフォームが実際に書き込む対象フィールドの
**現在の DB 値** から指紋（短縮 sha256）を作り、フォーム描画時に hidden で渡し、
保存時に再計算して照合する。

行バージョン列（`version_id_col`）を使わない理由:
  - `characters` はバッチ・実行時処理（chronicle の self_history、gate の away_until 等）
    でも更新されるため、行単位のバージョンでは誤検知だらけになる。
  - `global_settings` は `scheduler_heartbeat_*` などのランタイム値と同居しており、
    行／テーブル単位のバージョンが意味を成さない。
フォームが書き込む対象フィールドだけを見れば、「他端末が同じ項目を変えた」場合
だけを検出できる。

条件付きでしか書き込まないフィールド（画像のように「送られたときだけ上書き」する
もの）は対象に含めない。送らなければ他端末の値を壊さない＝先祖返りしないため。
"""

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from backend.lib.initiative_budget import CAP_SETTING_KEY

#: フォームが指紋を往復させる hidden フィールド名。
FINGERPRINT_FIELD = "_fp"

#: 衝突を承知で上書きするときに立てるフラグのフィールド名。
FORCE_FIELD = "_fp_force"

#: キャラクター編集フォームが無条件に上書きするカラム。
#: image_data は「新規アップロード時のみ更新」なので対象外。
CHARACTER_FIELDS = (
    "name",
    "system_prompt_block1",
    "ghost_model",
    "judge_preset_id",
    "user_label",
    "user_position",
    "user_visibility_note",
    "face_to_face_mode",
    "action_menu",
    "availability_schedule",
    "living_schedule_enabled",
    "speak_later_enabled",
    "bubble_color",
    "face_to_face_bg_images",
)

#: キャラクター編集フォームに同梱された うつつ（生活世界）設定が書き込む
#: シナリオ側のカラム（_persist_usual_world 参照）。
USUAL_SCENARIO_FIELDS = (
    "scenario",
    "pc_slots",
    "usual_config",
    "history_max_turns",
    "history_max_chars",
)

#: シナリオ編集フォームが無条件に上書きするカラム（banner_data は条件付きのため対象外）。
SCENARIO_FIELDS = (
    "title",
    "scenario",
    "intro",
    "history_max_turns",
    "history_max_chars",
    "custom_system_prompt",
    "dice_pool_spec",
    "pc_slots",
)

#: NPC 編集フォームが無条件に上書きするカラム（image_data は条件付きのため対象外）。
NPC_FIELDS = ("name", "description", "bubble_color")

#: 設定ページ「一般」フォームが書き込む設定キー。
#: API キー類はマスク値（●のみ）だと保存をスキップする＝他端末の値を壊さないため含めない。
GENERAL_SETTING_KEYS = (
    "user_name",
    "chronicle_time",
    "enable_time_awareness",
    "context_window_max_chronicled",
    CAP_SETTING_KEY,
    "translation_preset_id",
    "ollama_base_url",
    "ollama_no_think",
)

#: 設定ページ「embedding」フォームが書き込む設定キー。
EMBEDDING_SETTING_KEYS = ("embedding_provider", "embedding_model", "infinity_base_url")

#: 衝突時にユーザへ見せる文言。端末をまたいだ編集であることを名指しする。
CONFLICT_MESSAGE = (
    "この設定は別の端末（または別のタブ）で変更されている。"
    "このまま保存すると、そちらの変更が巻き戻る。"
)


def compute(values: Mapping[str, Any]) -> str:
    """対象フィールドの現在値から指紋を作る。

    値は JSON へ正規化してからハッシュする（dict のキー順・非 ASCII に依存しない）。
    datetime など JSON 化できない型は str() へ落とす。
    """
    payload = json.dumps(values, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _extract(obj: Any, fields: Iterable[str], prefix: str = "") -> dict[str, Any]:
    """ORM オブジェクトから対象フィールドを抜き出す。obj が None なら全て None 扱い。"""
    return {f"{prefix}{f}": (getattr(obj, f, None) if obj is not None else None) for f in fields}


def character_fingerprint(sqlite, character_id: str) -> str:
    """キャラクター編集フォームの指紋。同フォームが書く うつつ設定も込みで見る。"""
    values = _extract(sqlite.get_character(character_id), CHARACTER_FIELDS)
    values.update(
        _extract(sqlite.get_usual_scenario(character_id), USUAL_SCENARIO_FIELDS, prefix="usual.")
    )
    return compute(values)


def scenario_fingerprint(sqlite, scenario_id: str) -> str:
    """シナリオテンプレート編集フォームの指紋。"""
    return compute(_extract(sqlite.get_scenario(scenario_id), SCENARIO_FIELDS))


def npc_fingerprint(sqlite, npc_id: str) -> str:
    """NPC 編集フォーム（NPC 1 体につき 1 フォーム）の指紋。"""
    return compute(_extract(sqlite.get_scenario_npc(npc_id), NPC_FIELDS))


def npc_fingerprints(sqlite, npcs: Iterable[Any]) -> dict[str, str]:
    """NPC 一覧から {npc_id: 指紋} を作る（テンプレートが hidden へ埋めるため）。"""
    return {n.id: compute(_extract(n, NPC_FIELDS)) for n in npcs or []}


def settings_fingerprint(sqlite, keys: Iterable[str]) -> str:
    """グローバル設定のうち、指定キー群だけを見た指紋。"""
    stored = sqlite.get_all_settings()
    return compute({k: stored.get(k) for k in keys})


def is_forced(form) -> bool:
    """「衝突を承知で上書きする」フラグが立っているか。"""
    return bool(form.get(FORCE_FIELD))


def verify(form, current: str) -> bool:
    """フォームが持つ指紋が現在の DB 状態と一致するか。

    指紋が送られてこないフォーム（hidden を持たない別経路・古いページ）は
    素通しする。楽観ロックは「巻き戻りを検出する」ための仕組みであって、
    保存経路を塞ぐためのものではない。
    """
    if is_forced(form):
        return True
    submitted = (form.get(FINGERPRINT_FIELD) or "").strip()
    if not submitted:
        return True
    return submitted == current
