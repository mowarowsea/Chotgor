"""楽観ロック（設定フォームの先祖返り防止）のテスト。

スマホと PC で同じ編集画面を開いたまま自動保存が走ると、古い値を抱えたフォームが
全フィールドを丸ごと送って他端末の変更を巻き戻す（先祖返り）。
backend/lib/optimistic_lock.py は「そのフォームが書き込む項目の現在値」から
指紋を作り、フォーム描画時に hidden で渡して保存時に照合することでこれを検出する。

ここでは 2 層を検証する:
  1. 指紋の算出・照合規則（フォームが書かない項目に反応しないこと が肝）
  2. 管理UI 4 フォーム（キャラクター / シナリオ / NPC / 設定）の実際の 409 挙動
"""

import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from backend.api.ui import common as ui_common
from backend.api.ui.characters import router as characters_router
from backend.api.ui.scenarios import router as scenarios_router
from backend.api.ui.settings import router as settings_router
from backend.lib.optimistic_lock import (
    EMBEDDING_SETTING_KEYS,
    GENERAL_SETTING_KEYS,
    character_fingerprint,
    npc_fingerprint,
    scenario_fingerprint,
    settings_fingerprint,
    verify,
)

_TEMPLATES_DIR = str(Path(__file__).parent.parent / "backend" / "templates")

#: AJAX（自動保存）としてリクエストすることを示すヘッダ。
AJAX = {"X-Requested-With": "fetch"}


@pytest.fixture
def ui_client(sqlite_store, monkeypatch):
    """管理UI の編集系ルーターと実テンプレートを組み込んだテストクライアント。

    テンプレートは本物（backend/templates）を使い、hidden の埋め込み漏れや
    衝突ページ（conflict.html）の Jinja 構文崩れもテストで検出できるようにする。
    リダイレクトは追わない（303 / 409 の区別をそのまま見るため）。
    """
    monkeypatch.setattr(ui_common, "templates", Jinja2Templates(directory=_TEMPLATES_DIR))
    app = FastAPI()
    app.include_router(characters_router)
    app.include_router(scenarios_router)
    app.include_router(settings_router)
    app.state.sqlite = sqlite_store
    return TestClient(app, follow_redirects=False)


@pytest.fixture
def character(sqlite_store):
    """指紋の対象になるフィールドを一通り持ったキャラクターを1体作る。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(
        character_id=char_id,
        name="はる",
        system_prompt_block1="ここに人格が書かれている",
        user_label="もわ",
        user_position="同居人",
    )
    return sqlite_store.get_character(char_id)


@pytest.fixture
def scenario(sqlite_store):
    """NPC を1体ぶら下げたシナリオテンプレートを作る。"""
    scenario_id = str(uuid.uuid4())
    sqlite_store.create_scenario(
        scenario_id=scenario_id,
        title="古城に潜む影",
        scenario="夜の古城。",
    )
    sqlite_store.create_scenario_npc(
        npc_id=str(uuid.uuid4()),
        scenario_id=scenario_id,
        name="門番",
        description="無口。",
    )
    return sqlite_store.get_scenario(scenario_id)


class TestFingerprintComputation:
    """指紋の算出規則のテスト。

    設計の肝は「フォームが無条件に書き込む項目だけを見る」こと。
    characters はバッチ（chronicle の self_history）や実行時処理（gate の away_until）
    でも更新されるため、行全体のバージョンだと誤検知だらけになる。
    ここではその境界（見る項目 / 見ない項目）が意図どおりかを確かめる。
    """

    def test_same_state_yields_same_fingerprint(self, sqlite_store, character):
        """DB が変化していなければ、何度計算しても同じ指紋になること。"""
        first = character_fingerprint(sqlite_store, character.id)
        second = character_fingerprint(sqlite_store, character.id)

        assert first == second

    def test_watched_field_change_moves_fingerprint(self, sqlite_store, character):
        """フォームが書く項目（人格テキスト）が変われば指紋も変わること。"""
        before = character_fingerprint(sqlite_store, character.id)

        sqlite_store.update_character(character.id, system_prompt_block1="別端末で書き換えた")

        assert character_fingerprint(sqlite_store, character.id) != before

    def test_batch_updated_fields_do_not_move_fingerprint(self, sqlite_store, character):
        """chronicle / gate が触る項目では指紋が動かないこと（誤検知の抑止）。

        self_history・relationship_state は chronicle が、away_until は
        availability ゲートが更新する。いずれも編集フォームは書かないので、
        これらで 409 を出すのは純粋な誤検知になる。
        """
        from datetime import datetime, timedelta

        before = character_fingerprint(sqlite_store, character.id)

        sqlite_store.update_character(
            character.id,
            self_history="バッチが書いた経緯",
            relationship_state="バッチが書いた関係",
            away_until=datetime.now() + timedelta(hours=1),
            away_reason="疲労",
        )

        assert character_fingerprint(sqlite_store, character.id) == before

    def test_image_change_does_not_move_fingerprint(self, sqlite_store, character):
        """画像は「送られたときだけ上書き」なので指紋の対象外であること。

        フォームが画像を送らなければ他端末の画像は壊れない＝先祖返りしないため、
        画像差し替えを衝突として扱うと保存を止める理由のない 409 になる。
        """
        before = character_fingerprint(sqlite_store, character.id)

        sqlite_store.update_character(character.id, image_data="data:image/png;base64,AAAA")

        assert character_fingerprint(sqlite_store, character.id) == before

    def test_usual_world_is_part_of_character_fingerprint(self, sqlite_store, character):
        """同じフォームが書く うつつ設定の変化も、キャラの指紋に含まれること。

        うつつ（生活世界）はキャラ編集フォームに同梱され、別テーブル（scenarios）へ
        保存される。指紋がキャラ行しか見ていないと、うつつ側だけの先祖返りを取り逃す。
        """
        sqlite_store.create_scenario(
            scenario_id=str(uuid.uuid4()),
            title="はる のうつつ",
            owner_character_id=character.id,
            usual_config={"enabled": True, "slots": ["09:00"]},
        )
        before = character_fingerprint(sqlite_store, character.id)

        usual = sqlite_store.get_usual_scenario(character.id)
        sqlite_store.update_scenario(usual.id, usual_config={"enabled": True, "slots": ["21:00"]})

        assert character_fingerprint(sqlite_store, character.id) != before

    def test_settings_fingerprint_ignores_runtime_keys(self, sqlite_store):
        """global_settings のランタイム値では設定画面の指紋が動かないこと。

        global_settings には scheduler_heartbeat_* などの実行時カウンタが同居している。
        テーブル単位のバージョンではなくキー限定の指紋にしている理由がこれ。
        """
        sqlite_store.set_setting("user_name", "もわ")
        before = settings_fingerprint(sqlite_store, GENERAL_SETTING_KEYS)

        sqlite_store.set_setting("scheduler_heartbeat_usual", "2026-08-19T12:00:00")

        assert settings_fingerprint(sqlite_store, GENERAL_SETTING_KEYS) == before

    def test_settings_forms_have_independent_fingerprints(self, sqlite_store):
        """一般設定と embedding 設定は別フォーム＝互いの変更で衝突しないこと。"""
        before = settings_fingerprint(sqlite_store, EMBEDDING_SETTING_KEYS)

        sqlite_store.set_setting("user_name", "別の呼び名")

        assert settings_fingerprint(sqlite_store, EMBEDDING_SETTING_KEYS) == before


class TestVerify:
    """指紋の照合規則のテスト。

    verify は「巻き戻りを検出する」ためのもので、保存経路を塞ぐためのものではない。
    指紋を持たないフォーム（hidden を持たない別経路・古いページ）は素通しし、
    強制フラグは常に通す、という 2 つの逃げ道が仕様として要る。
    """

    def test_matching_fingerprint_passes(self):
        """フォームの指紋が現在値と一致すれば通ること。"""
        assert verify({"_fp": "abc123"}, "abc123") is True

    def test_stale_fingerprint_is_rejected(self):
        """フォームの指紋が古ければ弾かれること。"""
        assert verify({"_fp": "old"}, "new") is False

    def test_missing_fingerprint_passes_through(self):
        """指紋を送ってこないフォームは素通しすること（保存経路を塞がない）。"""
        assert verify({}, "current") is True
        assert verify({"_fp": "  "}, "current") is True

    def test_force_flag_overrides_mismatch(self):
        """強制フラグが立っていれば、指紋がずれていても通すこと。"""
        assert verify({"_fp": "old", "_fp_force": "1"}, "new") is True


class TestCharacterFormOptimisticLock:
    """キャラクター編集フォーム（自動保存）の先祖返り防止のテスト。

    このフォームは 1 フィールドの変更ごとに全項目を POST するため、
    放置したタブがそのまま「他端末の変更を巻き戻す装置」になりうる。
    """

    def test_edit_page_embeds_fingerprint(self, ui_client, sqlite_store, character):
        """編集ページの HTML に、現在値の指紋が hidden として埋まっていること。"""
        res = ui_client.get(f"/ui/characters/{character.id}")

        assert res.status_code == 200
        assert f'name="_fp" value="{character_fingerprint(sqlite_store, character.id)}"' in res.text

    def test_save_with_current_fingerprint_succeeds(self, ui_client, sqlite_store, character):
        """指紋が最新なら通常どおり保存されること。"""
        fp = character_fingerprint(sqlite_store, character.id)

        res = ui_client.post(
            f"/ui/characters/{character.id}",
            data={"name": "はる", "system_prompt_block1": "自分の端末で書いた", "_fp": fp},
            headers=AJAX,
        )

        assert res.status_code == 200
        assert res.json()["ok"] is True
        assert sqlite_store.get_character(character.id).system_prompt_block1 == "自分の端末で書いた"

    def test_stale_form_does_not_roll_back_other_device(self, ui_client, sqlite_store, character):
        """古いフォームからの自動保存が 409 になり、他端末の変更が残ること。"""
        stale_fp = character_fingerprint(sqlite_store, character.id)
        # 別端末での編集
        sqlite_store.update_character(character.id, system_prompt_block1="PCで書き足した")

        res = ui_client.post(
            f"/ui/characters/{character.id}",
            data={"name": "はる", "system_prompt_block1": "スマホが抱えていた古い本文", "_fp": stale_fp},
            headers=AJAX,
        )

        assert res.status_code == 409
        assert res.json()["conflict"] is True
        assert sqlite_store.get_character(character.id).system_prompt_block1 == "PCで書き足した"

    def test_forced_save_overwrites(self, ui_client, sqlite_store, character):
        """強制フラグ付きなら、衝突を承知で上書きできること。"""
        stale_fp = character_fingerprint(sqlite_store, character.id)
        sqlite_store.update_character(character.id, system_prompt_block1="PCで書き足した")

        res = ui_client.post(
            f"/ui/characters/{character.id}",
            data={
                "name": "はる",
                "system_prompt_block1": "スマホの内容で上書きする",
                "_fp": stale_fp,
                "_fp_force": "1",
            },
            headers=AJAX,
        )

        assert res.status_code == 200
        assert sqlite_store.get_character(character.id).system_prompt_block1 == "スマホの内容で上書きする"

    def test_response_carries_refreshed_fingerprint(self, ui_client, sqlite_store, character):
        """保存レスポンスが新しい指紋を返し、連続保存が自分自身と衝突しないこと。

        自動保存は 1 フィールドごとに走るため、保存のたびに指紋が更新されないと
        2 回目以降が必ず 409 になってしまう。
        """
        fp = character_fingerprint(sqlite_store, character.id)

        first = ui_client.post(
            f"/ui/characters/{character.id}",
            data={"name": "はる", "system_prompt_block1": "1回目", "_fp": fp},
            headers=AJAX,
        )
        refreshed = first.json()["fp"]
        # 返された指紋が「保存直後の DB 状態」を表していること
        assert refreshed == character_fingerprint(sqlite_store, character.id)

        second = ui_client.post(
            f"/ui/characters/{character.id}",
            data={"name": "はる", "system_prompt_block1": "2回目", "_fp": refreshed},
            headers=AJAX,
        )

        assert second.status_code == 200
        assert sqlite_store.get_character(character.id).system_prompt_block1 == "2回目"

    def test_non_ajax_conflict_renders_confirmation_page(self, ui_client, sqlite_store, character):
        """通常のフォーム送信で衝突したら、上書き確認ページが返ること。

        JS 無しの経路でも入力内容を失わずに選び直せる必要がある。
        """
        stale_fp = character_fingerprint(sqlite_store, character.id)
        sqlite_store.update_character(character.id, system_prompt_block1="PCで書き足した")

        res = ui_client.post(
            f"/ui/characters/{character.id}",
            data={"name": "はる", "system_prompt_block1": "古い本文", "_fp": stale_fp},
        )

        assert res.status_code == 409
        assert "このまま上書きする" in res.text
        # 送信内容が hidden として保持され、強制フラグ付きで再送できること
        assert 'value="古い本文"' in res.text
        assert 'name="_fp_force"' in res.text


class TestScenarioFormOptimisticLock:
    """シナリオテンプレート編集フォームの先祖返り防止のテスト。"""

    def test_edit_page_embeds_fingerprints(self, ui_client, sqlite_store, scenario):
        """シナリオ本体と NPC の指紋が、それぞれのフォームに埋まっていること。

        NPC は 1 体ごとに独立したフォーム（モーダル）なので指紋も個別に持つ。
        """
        npc = sqlite_store.list_scenario_npcs(scenario.id)[0]

        res = ui_client.get(f"/ui/scenarios/{scenario.id}/edit")

        assert res.status_code == 200
        assert f'name="_fp" value="{scenario_fingerprint(sqlite_store, scenario.id)}"' in res.text
        assert f'name="_fp" value="{npc_fingerprint(sqlite_store, npc.id)}"' in res.text

    def test_stale_form_does_not_roll_back_other_device(self, ui_client, sqlite_store, scenario):
        """古いフォームからの保存が 409 になり、他端末の変更が残ること。"""
        stale_fp = scenario_fingerprint(sqlite_store, scenario.id)
        sqlite_store.update_scenario(scenario.id, scenario="PCで書き直した舞台")

        res = ui_client.post(
            f"/ui/scenarios/{scenario.id}/edit",
            data={"title": "古城に潜む影", "scenario": "スマホの古い舞台", "_fp": stale_fp},
            headers=AJAX,
        )

        assert res.status_code == 409
        assert sqlite_store.get_scenario(scenario.id).scenario == "PCで書き直した舞台"

    def test_current_fingerprint_saves_and_refreshes(self, ui_client, sqlite_store, scenario):
        """最新の指紋なら保存でき、新しい指紋が返ること。"""
        fp = scenario_fingerprint(sqlite_store, scenario.id)

        res = ui_client.post(
            f"/ui/scenarios/{scenario.id}/edit",
            data={"title": "古城に潜む影", "scenario": "書き換えた舞台", "_fp": fp},
            headers=AJAX,
        )

        assert res.status_code == 200
        assert res.json()["fp"] == scenario_fingerprint(sqlite_store, scenario.id)
        assert sqlite_store.get_scenario(scenario.id).scenario == "書き換えた舞台"


class TestNpcFormOptimisticLock:
    """NPC 編集フォームの先祖返り防止のテスト。"""

    def test_stale_form_does_not_roll_back_other_device(self, ui_client, sqlite_store, scenario):
        """別端末で変えた NPC 説明を、古いフォームが巻き戻さないこと。"""
        npc = sqlite_store.list_scenario_npcs(scenario.id)[0]
        stale_fp = npc_fingerprint(sqlite_store, npc.id)
        sqlite_store.update_scenario_npc(npc.id, description="PCで書き足した設定")

        res = ui_client.post(
            f"/ui/scenarios/{scenario.id}/npcs/{npc.id}/edit",
            data={"name": "門番", "description": "スマホの古い設定", "_fp": stale_fp},
            headers=AJAX,
        )

        assert res.status_code == 409
        assert sqlite_store.get_scenario_npc(npc.id).description == "PCで書き足した設定"

    def test_other_npc_edit_does_not_block_this_npc(self, ui_client, sqlite_store, scenario):
        """別の NPC が編集されても、この NPC のフォームは衝突しないこと。

        指紋が NPC 単位でなくシナリオ単位だと、無関係な編集で保存が止まる。
        """
        npc = sqlite_store.list_scenario_npcs(scenario.id)[0]
        fp = npc_fingerprint(sqlite_store, npc.id)
        sqlite_store.create_scenario_npc(
            npc_id=str(uuid.uuid4()),
            scenario_id=scenario.id,
            name="行商人",
            description="別端末で足された NPC",
        )

        res = ui_client.post(
            f"/ui/scenarios/{scenario.id}/npcs/{npc.id}/edit",
            data={"name": "門番", "description": "こちらで書いた設定", "_fp": fp},
            headers=AJAX,
        )

        assert res.status_code == 200
        assert sqlite_store.get_scenario_npc(npc.id).description == "こちらで書いた設定"


class TestSettingsFormOptimisticLock:
    """設定ページ（一般 / embedding）の先祖返り防止のテスト。"""

    def test_settings_page_embeds_both_fingerprints(self, ui_client, sqlite_store):
        """一般設定と embedding 設定、それぞれの指紋が埋まっていること。"""
        res = ui_client.get("/ui/settings")

        assert res.status_code == 200
        assert f'name="_fp" value="{settings_fingerprint(sqlite_store, GENERAL_SETTING_KEYS)}"' in res.text
        assert f'name="_fp" value="{settings_fingerprint(sqlite_store, EMBEDDING_SETTING_KEYS)}"' in res.text

    def test_stale_form_does_not_roll_back_other_device(self, ui_client, sqlite_store):
        """古いフォームからの一般設定保存が 409 になり、他端末の値が残ること。"""
        stale_fp = settings_fingerprint(sqlite_store, GENERAL_SETTING_KEYS)
        sqlite_store.set_setting("user_name", "PCで変えた呼び名")

        res = ui_client.post(
            "/ui/settings/general",
            data={"user_name": "スマホの古い呼び名", "_fp": stale_fp},
            headers=AJAX,
        )

        assert res.status_code == 409
        assert sqlite_store.get_setting("user_name") == "PCで変えた呼び名"

    def test_api_key_entry_does_not_trigger_conflict(self, ui_client, sqlite_store):
        """API キーを別端末で変えても、一般設定フォームが衝突しないこと。

        API キーはマスク値（●のみ）だと保存をスキップする＝古いフォームでも
        巻き戻らないため、指紋の対象から外してある。
        """
        fp = settings_fingerprint(sqlite_store, GENERAL_SETTING_KEYS)
        sqlite_store.set_setting("anthropic_api_key", "sk-ant-別端末で設定")

        res = ui_client.post(
            "/ui/settings/general",
            data={"user_name": "もわ", "anthropic_api_key": "●●●●", "_fp": fp},
            headers=AJAX,
        )

        assert res.status_code == 200
        assert sqlite_store.get_setting("anthropic_api_key") == "sk-ant-別端末で設定"
        assert sqlite_store.get_setting("user_name") == "もわ"

    def test_embedding_form_conflict_blocks_reindex(self, ui_client, sqlite_store):
        """embedding 設定の衝突では、保存も再インデックスも走らないこと。

        巻き戻ると全記憶が意図しないモデルで焼き直されるため、
        ここでの先祖返りは他フォームより高くつく。
        """
        sqlite_store.set_setting("embedding_provider", "infinity")
        stale_fp = settings_fingerprint(sqlite_store, EMBEDDING_SETTING_KEYS)
        sqlite_store.set_setting("embedding_model", "cl-nagoya/ruri-v3-310m")

        res = ui_client.post(
            "/ui/settings/embedding",
            data={
                "embedding_provider": "google",
                "embedding_model": "gemini-embedding-001",
                "infinity_base_url": "http://localhost:7997",
                "_fp": stale_fp,
            },
        )

        assert res.status_code == 409
        assert sqlite_store.get_setting("embedding_provider") == "infinity"
        assert sqlite_store.get_setting("embedding_model") == "cl-nagoya/ruri-v3-310m"
