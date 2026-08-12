"""チャットバブルの配色スロット（bubble_color）のテスト。

同席するキャラクター同士でバブル色が偶然かぶったときに、設定画面から手で
振り分けられるようにするための列。検証する観点:

    - フォーム値の正規化（`_parse_bubble_color`）— 空欄・範囲外・非数値は「自動」に落ちる
    - スウォッチUIの「使用中」マップ（`_bubble_color_owners`）— 自動配色の相手は載らない
    - 永続化 — キャラクター / シナリオNPC いずれも保存・更新・None への戻しができる
    - API シリアライズ — フロントが色を引けるようレスポンスに含まれる
"""

import uuid

import pytest

from backend.api.ui.common import _bubble_color_owners, _parse_bubble_color
from backend.api.utils import char_to_dict
from backend.services.scenario_chat.serializers import scenario_npc_to_dict
from tests._scenario_sqlite_helpers import _make_npc, _make_scenario


class TestParseBubbleColor:
    """フォームの `bubble_color` 文字列を int|None へ正規化する処理を検証する。

    「自動」は空文字で送られてくる。不正値でフォーム保存が落ちるより、
    自動配色（None）へ倒すほうが実害が小さいのでその方針を固定する。
    """

    @pytest.mark.parametrize("raw", ["0", "4", "9"])
    def test_valid_indices(self, raw):
        """パレット範囲内の数値はそのまま int になること。"""
        assert _parse_bubble_color({"bubble_color": raw}) == int(raw)

    @pytest.mark.parametrize("raw", ["", "   ", None])
    def test_blank_means_auto(self, raw):
        """空欄・未送信は「自動」= None になること。"""
        assert _parse_bubble_color({"bubble_color": raw}) is None

    def test_missing_key_means_auto(self):
        """キー自体が無いフォームでも例外にならず None になること。"""
        assert _parse_bubble_color({}) is None

    @pytest.mark.parametrize("raw", ["-1", "10", "999", "abc", "3.5"])
    def test_out_of_range_or_garbage_means_auto(self, raw):
        """範囲外・非数値は握りつぶして None にすること（保存を失敗させない）。"""
        assert _parse_bubble_color({"bubble_color": raw}) is None


class TestBubbleColorOwners:
    """「この色は誰かが使用中」を示すためのマップ生成を検証する。"""

    class _Item:
        """name / bubble_color だけを持つ NPC・キャラクター風のダミー。"""

        def __init__(self, name, bubble_color):
            self.name = name
            self.bubble_color = bubble_color

    def test_groups_names_by_color(self):
        """同じ色を選んだ相手は 1 つの色番号にまとまること。"""
        owners = _bubble_color_owners(
            [self._Item("紅音", 4), self._Item("冷華", 7), self._Item("はる", 4)]
        )
        assert owners == {4: ["紅音", "はる"], 7: ["冷華"]}

    def test_auto_entries_are_omitted(self):
        """自動配色（None）は色が確定しないのでマップに載せないこと。"""
        assert _bubble_color_owners([self._Item("紅音", None)]) == {}

    def test_empty_input(self):
        """空リスト・None を渡しても空マップを返すこと。"""
        assert _bubble_color_owners([]) == {}
        assert _bubble_color_owners(None) == {}


class TestScenarioNpcBubbleColor:
    """シナリオNPC の bubble_color 永続化を検証する。"""

    def test_default_is_auto(self, sqlite_store):
        """未指定なら None（自動配色）で作成されること。既存NPCの見た目を変えないため。"""
        scenario = _make_scenario(sqlite_store)
        npc = _make_npc(sqlite_store, scenario.id, name="紅音")
        assert npc.bubble_color is None

    def test_create_with_color(self, sqlite_store):
        """作成時に指定した色が保存されること。"""
        scenario = _make_scenario(sqlite_store)
        npc = _make_npc(sqlite_store, scenario.id, name="紅音", bubble_color=4)
        assert sqlite_store.get_scenario_npc(npc.id).bubble_color == 4

    def test_update_and_reset_to_auto(self, sqlite_store):
        """更新で色を付け替えられ、None を渡せば自動配色へ戻せること。"""
        scenario = _make_scenario(sqlite_store)
        npc = _make_npc(sqlite_store, scenario.id, name="冷華", bubble_color=4)
        sqlite_store.update_scenario_npc(npc.id, bubble_color=8)
        assert sqlite_store.get_scenario_npc(npc.id).bubble_color == 8
        sqlite_store.update_scenario_npc(npc.id, bubble_color=None)
        assert sqlite_store.get_scenario_npc(npc.id).bubble_color is None

    def test_same_color_allowed_for_two_npcs(self, sqlite_store):
        """同じ色を複数NPCに設定できること（UI は注意喚起のみで禁止はしない）。"""
        scenario = _make_scenario(sqlite_store)
        a = _make_npc(sqlite_store, scenario.id, name="紅音", bubble_color=4)
        b = _make_npc(sqlite_store, scenario.id, name="冷華", bubble_color=4)
        assert sqlite_store.get_scenario_npc(a.id).bubble_color == 4
        assert sqlite_store.get_scenario_npc(b.id).bubble_color == 4

    def test_serializer_includes_color(self, sqlite_store):
        """API レスポンスに bubble_color が含まれること（フロントの配色解決に必要）。"""
        scenario = _make_scenario(sqlite_store)
        npc = _make_npc(sqlite_store, scenario.id, name="紅音", bubble_color=2)
        assert scenario_npc_to_dict(npc)["bubble_color"] == 2


class TestCharacterBubbleColor:
    """キャラクター本体の bubble_color 永続化を検証する。"""

    def _make_character(self, store, name="はる"):
        return store.create_character(character_id=str(uuid.uuid4()), name=name)

    def test_default_is_auto(self, sqlite_store):
        """新規キャラクターは自動配色（None）で始まること。"""
        char = self._make_character(sqlite_store)
        assert char.bubble_color is None

    def test_update_and_reset_to_auto(self, sqlite_store):
        """更新で色を設定でき、None で自動配色へ戻せること。"""
        char = self._make_character(sqlite_store)
        sqlite_store.update_character(char.id, bubble_color=6)
        assert sqlite_store.get_character(char.id).bubble_color == 6
        sqlite_store.update_character(char.id, bubble_color=None)
        assert sqlite_store.get_character(char.id).bubble_color is None

    def test_serializer_includes_color(self, sqlite_store):
        """char_to_dict に bubble_color が含まれること。"""
        char = self._make_character(sqlite_store)
        sqlite_store.update_character(char.id, bubble_color=9)
        payload = char_to_dict(sqlite_store.get_character(char.id))
        assert payload["bubble_color"] == 9
