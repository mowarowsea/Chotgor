"""キャラクター・プリセット解決ヘルパー。

parse_model_id / resolve_character / resolve_preset / require_character / require_preset を
一元管理し、APIレイヤー全体での「名前 or UUID で検索 → なければ 404」の重複を排除する。
"""

from fastapi import HTTPException


def parse_model_id(model_id: str) -> tuple[str, str]:
    """{char_name}@{preset_name} 形式をパースして (char_name, preset_name) を返す。

    Args:
        model_id: "{char_name}@{preset_name}" 形式の文字列。

    Returns:
        (char_name, preset_name) のタプル。

    Raises:
        HTTPException 400: "@" が含まれない場合。
    """
    if "@" not in model_id:
        raise HTTPException(
            status_code=400,
            detail="model_id のフォーマットが不正です。'{char_name}@{preset_name}' 形式で指定してください",
        )
    return model_id.rsplit("@", 1)


def resolve_character(sqlite, identifier: str):
    """名前またはUUIDでキャラクターを取得する（名前優先）。

    見つからない場合は None を返す。HTTP例外は送出しない。
    """
    return sqlite.get_character_by_name(identifier) or sqlite.get_character(identifier)


def resolve_preset(sqlite, identifier: str):
    """名前またはUUIDでモデルプリセットを取得する（名前優先）。

    見つからない場合は None を返す。HTTP例外は送出しない。
    """
    return sqlite.get_model_preset_by_name(identifier) or sqlite.get_model_preset(identifier)


def require_character(sqlite, identifier: str):
    """名前またはUUIDでキャラクターを取得し、存在しない場合は 404 を送出する。

    Args:
        sqlite: SQLiteStore インスタンス。
        identifier: キャラクター名またはUUID。

    Returns:
        Character ORM オブジェクト。

    Raises:
        HTTPException 404: キャラクターが存在しない場合。
    """
    char = resolve_character(sqlite, identifier)
    if not char:
        raise HTTPException(status_code=404, detail=f"キャラクター '{identifier}' が見つかりません")
    return char


def require_preset(sqlite, identifier: str):
    """名前またはUUIDでプリセットを取得し、存在しない場合は 404 を送出する。

    Args:
        sqlite: SQLiteStore インスタンス。
        identifier: プリセット名またはUUID。

    Returns:
        LLMModelPreset ORM オブジェクト。

    Raises:
        HTTPException 404: プリセットが存在しない場合。
    """
    preset = resolve_preset(sqlite, identifier)
    if preset is None:
        raise HTTPException(status_code=404, detail=f"モデルプリセット '{identifier}' が見つかりません")
    return preset


def require_model_config(character, preset):
    """キャラクターに対してプリセットが有効化されているか検証し、設定辞書を返す。

    Args:
        character: Character ORM オブジェクト。
        preset: LLMModelPreset ORM オブジェクト。

    Returns:
        enabled_providers[preset.id] の設定辞書。

    Raises:
        HTTPException 400: プリセットがキャラクターで有効化されていない場合。
    """
    model_config = (character.enabled_providers or {}).get(preset.id)
    if model_config is None:
        raise HTTPException(
            status_code=400,
            detail=f"プリセット '{preset.name}' はキャラクター '{character.name}' で有効化されていません",
        )
    return model_config
