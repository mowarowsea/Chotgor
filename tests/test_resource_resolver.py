from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from backend.api.resource_resolver import (
    parse_model_id,
    require_character,
    require_model_config,
    require_preset,
    resolve_character,
    resolve_preset,
)


def test_parse_model_id_splits_on_last_at_sign():
    assert parse_model_id("char@preset@v2") == ["char@preset", "v2"]


def test_parse_model_id_rejects_missing_separator():
    with pytest.raises(HTTPException) as exc_info:
        parse_model_id("char-only")

    assert exc_info.value.status_code == 400


def test_resolve_character_prefers_name_lookup():
    sqlite = Mock()
    by_name = SimpleNamespace(id="char-1", name="Momo")
    sqlite.get_character_by_name.return_value = by_name

    resolved = resolve_character(sqlite, "Momo")

    assert resolved is by_name
    sqlite.get_character.assert_not_called()


def test_resolve_character_falls_back_to_id_lookup():
    sqlite = Mock()
    by_id = SimpleNamespace(id="char-1", name="Momo")
    sqlite.get_character_by_name.return_value = None
    sqlite.get_character.return_value = by_id

    resolved = resolve_character(sqlite, "char-1")

    assert resolved is by_id
    sqlite.get_character.assert_called_once_with("char-1")


def test_require_character_raises_404_when_not_found():
    sqlite = Mock()
    sqlite.get_character_by_name.return_value = None
    sqlite.get_character.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        require_character(sqlite, "missing")

    assert exc_info.value.status_code == 404


def test_resolve_preset_prefers_name_lookup():
    sqlite = Mock()
    by_name = SimpleNamespace(id="preset-1", name="default")
    sqlite.get_model_preset_by_name.return_value = by_name

    resolved = resolve_preset(sqlite, "default")

    assert resolved is by_name
    sqlite.get_model_preset.assert_not_called()


def test_resolve_preset_falls_back_to_id_lookup():
    sqlite = Mock()
    by_id = SimpleNamespace(id="preset-1", name="default")
    sqlite.get_model_preset_by_name.return_value = None
    sqlite.get_model_preset.return_value = by_id

    resolved = resolve_preset(sqlite, "preset-1")

    assert resolved is by_id
    sqlite.get_model_preset.assert_called_once_with("preset-1")


def test_require_preset_raises_404_when_not_found():
    sqlite = Mock()
    sqlite.get_model_preset_by_name.return_value = None
    sqlite.get_model_preset.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        require_preset(sqlite, "missing")

    assert exc_info.value.status_code == 404


def test_require_model_config_returns_enabled_provider_config():
    character = SimpleNamespace(
        name="Momo",
        enabled_providers={"preset-1": {"additional_instructions": "be kind"}},
    )
    preset = SimpleNamespace(id="preset-1", name="default")

    config = require_model_config(character, preset)

    assert config == {"additional_instructions": "be kind"}


def test_require_model_config_raises_400_when_preset_not_enabled():
    character = SimpleNamespace(name="Momo", enabled_providers={})
    preset = SimpleNamespace(id="preset-1", name="default")

    with pytest.raises(HTTPException) as exc_info:
        require_model_config(character, preset)

    assert exc_info.value.status_code == 400
