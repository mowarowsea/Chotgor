"""シナリオチャット API の Pydantic リクエストスキーマ定義。"""

from pydantic import BaseModel, Field


class ScenarioCreate(BaseModel):
    """シナリオテンプレート作成リクエスト。

    GM プリセットはテンプレートではなくセッション単位（SessionStart.gm_preset_id）で
    指定するため、本リクエストには含めない。
    custom_system_prompt を設定するとGMシステムプロンプトをカスタマイズできる。
    dice_pool_spec / pc_slots は ensemble_pc エンジンのセッション専用フィールド。
    """

    title: str = Field(min_length=1)
    scenario: str | None = None
    intro: str | None = None
    history_max_turns: int | None = None
    history_max_chars: int | None = None
    custom_system_prompt: str | None = None
    dice_pool_spec: dict | None = None
    # 全PC（ユーザPC含む）を定義する単一ソース。旧 user_alias は廃止し、ユーザPCも
    # この pc_slots の 1 枠として表現する（セッション開始時に player_type="user" を割り当てる）。
    pc_slots: list[dict] | None = None  # [{slot_id, name, description}]


class ScenarioUpdate(BaseModel):
    """シナリオテンプレート更新リクエスト（部分更新）。"""

    title: str | None = None
    scenario: str | None = None
    intro: str | None = None
    history_max_turns: int | None = None
    history_max_chars: int | None = None
    custom_system_prompt: str | None = None
    dice_pool_spec: dict | None = None
    pc_slots: list[dict] | None = None


class NpcCreate(BaseModel):
    """NPC 追加リクエスト。

    description には人物像・口調・話し方を自由テキストで全部詰め込む。
    image_data はアバター画像の base64 data URI（オプション）。
    """

    name: str = Field(min_length=1)
    description: str | None = None
    image_data: str | None = None


class NpcUpdate(BaseModel):
    """NPC 更新リクエスト（部分更新）。"""

    name: str | None = None
    description: str | None = None
    image_data: str | None = None


class SessionStart(BaseModel):
    """プレイセッション起動リクエスト。

    gm_preset_id / synopsis_preset_id はそれぞれセッション単位で必須。
    同一シナリオから複数セッションを起動した際にそれぞれ別モデルを選べる設計で、
    `synopsis_preset_id` はあらすじ蒸留専用（レートリミット節約のため別モデル指定可能）。
    UI 上は両方明示指定させる（同じプリセットでもよい）。

    engine_type は "ensemble"（既存・GMのみ）または "ensemble_pc"（GM + PC配役）。
    "ensemble_pc" の場合、親シナリオの `pc_slots` の各 slot_id について
    pc_assignments を 1 件以上指定する。形式:
        [{"slot_id":"pc1","player_type":"user"|"character",
          "character_id":"...","preset_id":"..."}]
    """

    scenario_id: str = Field(min_length=1)
    gm_preset_id: str = Field(min_length=1)
    synopsis_preset_id: str = Field(min_length=1)
    title: str | None = None
    engine_type: str = "ensemble"
    pc_assignments: list[dict] | None = None


class SessionUpdate(BaseModel):
    """プレイセッション更新リクエスト（タイトル / status / GM モデル / あらすじモデル）。"""

    title: str | None = None
    status: str | None = None
    gm_preset_id: str | None = None
    synopsis_preset_id: str | None = None


class SynopsisUpdate(BaseModel):
    """あらすじ部分更新リクエスト。

    `auto` と `manual` はそれぞれ独立に更新できる（リクエストに含めなければ触らない）。
    自動更新フローと違い、本 API では `auto` も上書き可能（ユーザが捏造記述を
    削除・修正するため）。
    """

    auto: str | None = None
    manual: str | None = None


class SynopsisRegenerate(BaseModel):
    """あらすじ手動作成（強制蒸留）リクエスト。

    `synopsis_preset_id` を指定すると、その preset をセッションへ永続化（記憶）した上で
    蒸留に使う。フロントの「あらすじ作成モーダル」で選んだモデルが次回以降の既定にも
    なる。省略時はセッション既定の `synopsis_preset_id` を使う。
    """

    synopsis_preset_id: str | None = None


class StreamRequest(BaseModel):
    """ストリーム発火リクエスト。

    auto_advance=True なら「ユーザは無言で続きを促す」モード。
    content は無視され、user turn は保存されない（履歴に痕跡を残さない）。

    regenerate_request_id を指定すると、そのログエントリに追記する形で
    再生成ログをまとめる。フロントは再生成ボタン押下時のみ前ターンの
    log_request_id を渡す（過去ターン編集時は渡さない）。
    """

    content: str = ""
    auto_advance: bool = False
    regenerate_request_id: str | None = None


