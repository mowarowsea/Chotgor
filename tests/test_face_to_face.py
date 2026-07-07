"""対面モード + external_scenes（scene-wrap 構造）の振る舞いテスト。

検証する観点:
    - 新カラム（characters.face_to_face_mode / face_to_face_bg_image,
      chat_messages.face_to_face）が冪等migrationで追加され、ORM/Storeから扱える
    - create_chat_message が face_to_face フラグを正しく保存する
    - build_system_prompt が face_to_face=True で対面ブロックを差し込み、False で消す
    - うつつスケジューラが face_to_face_mode==1 のオーナーをスキップし、冪等キーを当日日付で立てる
    - resolve_since_dt が「最新 SCENE_CLOSE 時刻」を起点に取る（PC runner 実行時の
      「今シーンの GM ターン」を起点に使わないこと）
    - build_unified_pc_messages が <history> + <{シーン名}>...</{シーン名}> 形式の
      単一メッセージを返し、シーンタグ・話者タグが正しく振られる
    - うつつシーン区切りが SCENE_CLOSE 単位で進む（PC ターン経由で誤分割しない）
    - 1on1 同セッション内で face_to_face モード切替があれば別シーン扱い
    - TRPG の自分の PC 発話タグが `<{役名}@{キャラ名}>` 形式になる

Chotgor 思想:
    うつつ・1on1・TRPG・Group は同じ一本の世界軸上の連続した時間。シーン境界はタグで
    包んで示すだけ。「ハレ／ケ」「幕間」のフレーミングはシステムからは加えない。
"""

import asyncio
import json
import types
from datetime import datetime, timedelta

import backend.main as mainmod
import backend.services.scenario_chat.service as svc
from backend.services.chat.request_builder import (
    _FACE_TO_FACE_NOTICE,
    build_system_prompt,
)
from backend.services.scenario_chat.external_scenes import (
    Scene,
    SpeechTurn,
    _build_1on1_scenes,
    _build_trpg_scenes,
    _build_usual_scenes,
    _resolve_trpg_role_name,
    build_unified_pc_messages,
    collect_all_scenes,
    render_scenes_to_text,
    resolve_since_dt,
)

from tests._scenario_sqlite_helpers import _make_scenario


# ---------------------------------------------------------------------------
# 共通ヘルパ
# ---------------------------------------------------------------------------


def _new_chat_message(
    store, session_id, role, content, face_to_face=0, character_name=None, delivered=True,
):
    """create_chat_message のショートカット（テスト用にメッセージIDを自動生成）。

    delivered=False で「預かり（escrow・delivered_at=NULL）」のメッセージを作れる。
    """
    import uuid

    return store.create_chat_message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role=role,
        content=content,
        face_to_face=face_to_face,
        character_name=character_name,
        delivered=delivered,
    )


def _make_usual_session(store, character_id="char-haru", session_id="usess"):
    """テスト用に最低限のうつつ世界＋セッションを組む。"""
    store.create_character(character_id, "はる")
    store.create_model_preset("gm", "GM", "anthropic", "claude-x")
    scenario = _make_scenario(
        store,
        title="日常",
        owner_character_id=character_id,
        pc_slots=[{"slot_id": "pc1", "name": "はる", "description": ""}],
    )
    store.create_scenario_session(
        session_id=session_id,
        scenario_id=scenario.id,
        title=scenario.title,
        gm_preset_id="gm",
        synopsis_preset_id="gm",
        engine_type="usual_days",
        pc_assignments=[
            {"slot_id": "pc1", "player_type": "character", "character_id": character_id}
        ],
    )
    return scenario, session_id


# ---------------------------------------------------------------------------
# Phase 1 — スキーマ・カラム永続化
# ---------------------------------------------------------------------------


class TestFaceToFaceSchema:
    """対面モードに伴うカラムの存在と既定値・永続化を確認する。"""

    def test_character_defaults(self, sqlite_store):
        """新規キャラは face_to_face_mode=0 / face_to_face_bg_image=None で作られること。"""
        sqlite_store.create_character("char-1", "テスト")
        char = sqlite_store.get_character("char-1")
        assert getattr(char, "face_to_face_mode", 0) == 0
        assert getattr(char, "face_to_face_bg_image", None) is None

    def test_character_update_face_to_face(self, sqlite_store):
        """update_character で face_to_face_mode / face_to_face_bg_image を更新できる。"""
        sqlite_store.create_character("char-1", "テスト")
        sqlite_store.update_character(
            "char-1",
            face_to_face_mode=1,
            face_to_face_bg_image="data:image/png;base64,AAAA",
        )
        char = sqlite_store.get_character("char-1")
        assert char.face_to_face_mode == 1
        assert char.face_to_face_bg_image == "data:image/png;base64,AAAA"

    def test_chat_message_face_to_face_default(self, sqlite_store):
        """face_to_face を省略すると既定値 0 で保存される。"""
        sqlite_store.create_chat_session("s1", "テスト@p")
        msg = _new_chat_message(sqlite_store, "s1", "user", "やあ")
        assert getattr(msg, "face_to_face", 0) == 0

    def test_chat_message_face_to_face_persists(self, sqlite_store):
        """face_to_face=1 を渡すと 1 として永続化される。"""
        sqlite_store.create_chat_session("s1", "テスト@p")
        msg = _new_chat_message(sqlite_store, "s1", "user", "対面中", face_to_face=1)
        assert msg.face_to_face == 1
        fetched = sqlite_store.list_chat_messages("s1")
        assert fetched[0].face_to_face == 1


# ---------------------------------------------------------------------------
# Phase 2 — システムプロンプト ブロック注入
# ---------------------------------------------------------------------------


class TestFaceToFaceSystemPrompt:
    """build_system_prompt の face_to_face フラグでブロック注入が切り替わることを検証する。"""

    def test_block_appears_when_true(self):
        """face_to_face=True で対面注釈ブロックが system prompt に含まれる。"""
        prompt = build_system_prompt(
            character_system_prompt="テストキャラ",
            user_label="太郎",
            face_to_face=True,
        )
        assert "## いまは対面でユーザと向き合っています" in prompt
        assert "強く記憶に残るはず" in prompt
        assert _FACE_TO_FACE_NOTICE in prompt

    def test_block_absent_when_false(self):
        """face_to_face=False では対面ブロックが入らない。"""
        prompt = build_system_prompt(
            character_system_prompt="テストキャラ",
            user_label="太郎",
            face_to_face=False,
        )
        assert "対面でユーザと向き合っています" not in prompt

    def test_block_absent_by_default(self):
        """face_to_face を渡さないデフォルトでも対面ブロックは入らない。"""
        prompt = build_system_prompt(
            character_system_prompt="テストキャラ",
            user_label="太郎",
        )
        assert "対面でユーザと向き合っています" not in prompt


# ---------------------------------------------------------------------------
# Phase 3 — うつつスケジューラの対面スキップ
# ---------------------------------------------------------------------------


def _build_usual_world_for_face(store, character_id="char-haru"):
    """対面スキップテスト用の最小うつつ世界を組む。"""
    pid = "preset-usual"
    store.create_character(character_id, "はる")
    store.create_model_preset(pid, "うつつ用", "anthropic", "claude-x")
    return _make_scenario(
        store,
        title="はるの日常",
        owner_character_id=character_id,
        pc_slots=[{"slot_id": "pc1", "name": "はる", "description": "主人公"}],
        usual_config={
            "enabled": True,
            "slots": ["00:00"],
            "max_responses_per_scene": 2,
            "gm_preset_id": pid,
            "pc_preset_id": pid,
        },
    )


class TestUsualSchedulerSkipsOnFaceToFace:
    """対面モード中はうつつが発火しないこと（通過分は捨てる）を検証する。"""

    def _fake_app(self, sqlite_store):
        return types.SimpleNamespace(
            state=types.SimpleNamespace(
                sqlite=sqlite_store, chat_service=object(),
            )
        )

    def test_skips_when_owner_in_face_to_face(self, sqlite_store, monkeypatch):
        _build_usual_world_for_face(sqlite_store)
        sqlite_store.update_character("char-haru", face_to_face_mode=1)
        calls: list[str] = []

        async def fake_scene(**kwargs):
            calls.append(kwargs["session_id"])
            return {"saved_turn_ids": [], "fired_turns": 0, "scene_closed": True, "error": None}

        monkeypatch.setattr(svc, "run_usual_days_scene", fake_scene)
        asyncio.run(mainmod._run_due_usual_scenes(self._fake_app(sqlite_store)))
        assert calls == []

    def test_skip_sets_idempotency_key(self, sqlite_store, monkeypatch):
        _build_usual_world_for_face(sqlite_store)
        sqlite_store.update_character("char-haru", face_to_face_mode=1)

        async def fake_scene(**kwargs):
            return {"saved_turn_ids": [], "fired_turns": 0, "scene_closed": True, "error": None}

        monkeypatch.setattr(svc, "run_usual_days_scene", fake_scene)
        asyncio.run(mainmod._run_due_usual_scenes(self._fake_app(sqlite_store)))
        today = datetime.now().date().isoformat()
        key = "usual_days_last_run_char-haru_00:00"
        assert sqlite_store.get_setting(key, "") == today

    def test_does_not_skip_when_mode_off(self, sqlite_store, monkeypatch):
        _build_usual_world_for_face(sqlite_store)
        calls: list[str] = []

        async def fake_scene(**kwargs):
            calls.append(kwargs["session_id"])
            return {"saved_turn_ids": [], "fired_turns": 0, "scene_closed": True, "error": None}

        monkeypatch.setattr(svc, "run_usual_days_scene", fake_scene)
        asyncio.run(mainmod._run_due_usual_scenes(self._fake_app(sqlite_store)))
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Phase 4 — resolve_since_dt
# ---------------------------------------------------------------------------


class TestResolveSinceDt:
    """resolve_since_dt が「最新 SCENE_CLOSE > 最古うつつターン > 24h前」の順で起点を返す。"""

    def test_fallback_when_no_usual_at_all(self, sqlite_store):
        sqlite_store.create_character("char-haru", "はる")
        now = datetime(2026, 6, 23, 12, 0)
        since = resolve_since_dt(sqlite_store, "char-haru", now=now)
        assert since == now - timedelta(hours=24)

    def test_uses_earliest_turn_when_no_scene_close(self, sqlite_store):
        _make_usual_session(sqlite_store)
        sqlite_store.create_scenario_turn(
            turn_id="t1", session_id="usess", turn_index=0,
            speaker_type="narrator", speaker_id=None, speaker_name="Narrator",
            content="朝が来た", raw_response="朝が来た",
        )
        since = resolve_since_dt(sqlite_store, "char-haru")
        assert (datetime.now() - since) < timedelta(seconds=5)

    def test_uses_latest_scene_close_not_latest_turn(self, sqlite_store):
        """SCENE_CLOSE 後にさらに新しい usual ターンがあっても、起点は SCENE_CLOSE 側。"""
        import time
        _make_usual_session(sqlite_store)
        sqlite_store.create_scenario_turn(
            turn_id="t1", session_id="usess", turn_index=0,
            speaker_type="narrator", speaker_id=None, speaker_name="Narrator",
            content="夜が更けた。\n\n[SCENE_CLOSE]",
            raw_response="夜が更けた。\n\n[SCENE_CLOSE]",
        )
        close_time = datetime.now()
        time.sleep(0.05)
        sqlite_store.create_scenario_turn(
            turn_id="t2", session_id="usess", turn_index=1,
            speaker_type="narrator", speaker_id=None, speaker_name="Narrator",
            content="翌朝。", raw_response="翌朝。",
        )
        since = resolve_since_dt(sqlite_store, "char-haru")
        assert since <= close_time + timedelta(seconds=0.1)


# ---------------------------------------------------------------------------
# Phase 5 — シーン構築（_build_usual_scenes / _build_1on1_scenes / etc.）
# ---------------------------------------------------------------------------


class TestBuildUsualScenes:
    """うつつ scenario_turns がシーン単位（SCENE_CLOSE で分割）に正しく分解される。"""

    def test_single_scene_no_scene_close(self, sqlite_store):
        """SCENE_CLOSE が無ければ全ターンが 1 シーンに収まる。"""
        _make_usual_session(sqlite_store)
        sqlite_store.create_scenario_turn(
            turn_id="t1", session_id="usess", turn_index=0,
            speaker_type="narrator", speaker_id=None, speaker_name="Narrator",
            content="朝", raw_response="GM1",
        )
        sqlite_store.create_scenario_turn(
            turn_id="t2", session_id="usess", turn_index=1,
            speaker_type="pc", speaker_id="char-haru", speaker_name="はる",
            content="おはよう", raw_response="",
        )
        sqlite_store.create_scenario_turn(
            turn_id="t3", session_id="usess", turn_index=2,
            speaker_type="narrator", speaker_id=None, speaker_name="Narrator",
            content="出勤", raw_response="GM2",  # 異なる raw だが SCENE_CLOSE 無し → 同シーン
        )
        history = sqlite_store.list_scenario_turns("usess")
        scenes = _build_usual_scenes(history, "char-haru", "はる", "太郎", "Narrator")
        assert len(scenes) == 1
        assert scenes[0].scene_tag == "はるの日常"
        assert len(scenes[0].turns) == 3
        # speaker_tag マッピング
        assert scenes[0].turns[0].speaker_tag == "Narrator"
        assert scenes[0].turns[1].speaker_tag == "はる"  # 自分のPC
        assert scenes[0].turns[2].speaker_tag == "Narrator"

    def test_split_at_scene_close(self, sqlite_store):
        """SCENE_CLOSE 後、別 raw_response の GM ターンで次シーンに分割される。"""
        _make_usual_session(sqlite_store)
        sqlite_store.create_scenario_turn(
            turn_id="a1", session_id="usess", turn_index=0,
            speaker_type="narrator", speaker_id=None, speaker_name="Narrator",
            content="一日終了。[SCENE_CLOSE]", raw_response="GM1 [SCENE_CLOSE]",
        )
        sqlite_store.create_scenario_turn(
            turn_id="b1", session_id="usess", turn_index=1,
            speaker_type="narrator", speaker_id=None, speaker_name="Narrator",
            content="翌朝", raw_response="GM2",
        )
        history = sqlite_store.list_scenario_turns("usess")
        scenes = _build_usual_scenes(history, "char-haru", "はる", "太郎", "Narrator")
        assert len(scenes) == 2
        # 両方とも「はるの日常」タグ
        assert scenes[0].scene_tag == scenes[1].scene_tag == "はるの日常"

    def test_pc_turn_does_not_split(self, sqlite_store):
        """PC ターン（raw=""）が挟まっても、SCENE_CLOSE 無しなら同シーンに留まる。"""
        _make_usual_session(sqlite_store)
        sqlite_store.create_scenario_turn(
            turn_id="t1", session_id="usess", turn_index=0,
            speaker_type="narrator", speaker_id=None, speaker_name="Narrator",
            content="朝", raw_response="GM1",
        )
        sqlite_store.create_scenario_turn(
            turn_id="t2", session_id="usess", turn_index=1,
            speaker_type="pc", speaker_id="char-haru", speaker_name="はる",
            content="出勤する", raw_response="",
        )
        sqlite_store.create_scenario_turn(
            turn_id="t3", session_id="usess", turn_index=2,
            speaker_type="narrator", speaker_id=None, speaker_name="Narrator",
            content="夕方", raw_response="GM2",
        )
        history = sqlite_store.list_scenario_turns("usess")
        scenes = _build_usual_scenes(history, "char-haru", "はる", "太郎", "Narrator")
        assert len(scenes) == 1


class TestBuild1on1Scenes:
    """1on1 セッションがシーン化される。face_to_face モード切替で別シーンになる。"""

    def test_text_only(self, sqlite_store):
        sqlite_store.create_character("char-haru", "はる")
        sqlite_store.create_chat_session("s1", "はる@p")
        _new_chat_message(sqlite_store, "s1", "user", "やあ", face_to_face=0)
        _new_chat_message(sqlite_store, "s1", "character", "うん", face_to_face=0, character_name="はる")
        scenes = _build_1on1_scenes(
            sqlite_store, "はる", "太郎",
            datetime.now() - timedelta(hours=1), datetime.now() + timedelta(seconds=1),
        )
        assert len(scenes) == 1
        assert scenes[0].scene_tag == "太郎とのテキストのやり取り"
        assert [t.speaker_tag for t in scenes[0].turns] == ["太郎", "はる"]
        assert [t.content for t in scenes[0].turns] == ["やあ", "うん"]

    def test_mode_switch_splits_scene(self, sqlite_store):
        """同セッション内で face_to_face モード切替があれば別シーンに分割される。"""
        sqlite_store.create_character("char-haru", "はる")
        sqlite_store.create_chat_session("s1", "はる@p")
        _new_chat_message(sqlite_store, "s1", "user", "テキスト1", face_to_face=0)
        _new_chat_message(sqlite_store, "s1", "character", "返信1", face_to_face=0, character_name="はる")
        _new_chat_message(sqlite_store, "s1", "user", "対面1", face_to_face=1)
        _new_chat_message(sqlite_store, "s1", "character", "対面返信", face_to_face=1, character_name="はる")
        scenes = _build_1on1_scenes(
            sqlite_store, "はる", "太郎",
            datetime.now() - timedelta(hours=1), datetime.now() + timedelta(seconds=1),
        )
        assert len(scenes) == 2
        assert scenes[0].scene_tag == "太郎とのテキストのやり取り"
        assert scenes[1].scene_tag == "太郎との対面"

    def test_undelivered_message_excluded(self, sqlite_store):
        """預かり中（delivered_at=NULL）のメッセージは統合履歴に含めない。

        escrow の思想上「まだキャラの身に起きていない」出来事であり、ここから
        漏れるとうつつシーンの最中にキャラが未配達メッセージへ応答してしまう
        （実障害: 1on1 の預かりメッセージへの回答がうつつ側に記録された）。
        配達済みメッセージだけがシーンに載ることを確認する。
        """
        sqlite_store.create_character("char-haru", "はる")
        sqlite_store.create_chat_session("s1", "はる@p")
        _new_chat_message(sqlite_store, "s1", "user", "配達済みの発言", face_to_face=0)
        _new_chat_message(sqlite_store, "s1", "character", "うん", face_to_face=0, character_name="はる")
        # うつつシーン進行中に届いた預かりメッセージ（キャラ未読）
        _new_chat_message(sqlite_store, "s1", "user", "預かり中の発言", face_to_face=0, delivered=False)
        scenes = _build_1on1_scenes(
            sqlite_store, "はる", "太郎",
            datetime.now() - timedelta(hours=1), datetime.now() + timedelta(seconds=1),
        )
        assert len(scenes) == 1
        contents = [t.content for t in scenes[0].turns]
        assert contents == ["配達済みの発言", "うん"]

    def test_undelivered_message_included_after_delivery(self, sqlite_store):
        """配達（mark_messages_delivered）後は同じメッセージがシーンに載る。"""
        sqlite_store.create_character("char-haru", "はる")
        sqlite_store.create_chat_session("s1", "はる@p")
        msg = _new_chat_message(
            sqlite_store, "s1", "user", "預かり中の発言", face_to_face=0, delivered=False,
        )
        sqlite_store.mark_messages_delivered([msg.id])
        scenes = _build_1on1_scenes(
            sqlite_store, "はる", "太郎",
            datetime.now() - timedelta(hours=1), datetime.now() + timedelta(seconds=1),
        )
        assert len(scenes) == 1
        assert [t.content for t in scenes[0].turns] == ["預かり中の発言"]


class TestBuildTrpgScenes:
    """TRPG シーンの構築と PC タグの役名@キャラ名 表記。"""

    def test_role_at_char_tag(self, sqlite_store):
        """自分の PC 発話タグが `<{役名}@{キャラ名}>` 形式になる。"""
        sqlite_store.create_character("char-haru", "はる")
        sqlite_store.create_model_preset("gm", "GM", "anthropic", "claude-x")
        scenario = _make_scenario(
            sqlite_store, title="ダンジョン編",
            pc_slots=[{"slot_id": "pc1", "name": "アリサ", "description": ""}],
        )
        sqlite_store.create_scenario_session(
            session_id="sess-1", scenario_id=scenario.id, title=scenario.title,
            gm_preset_id="gm", synopsis_preset_id="gm", engine_type="ensemble_pc",
            pc_assignments=[{"slot_id": "pc1", "player_type": "character", "character_id": "char-haru"}],
        )
        sqlite_store.create_scenario_turn(
            turn_id="t1", session_id="sess-1", turn_index=0,
            speaker_type="narrator", speaker_id=None, speaker_name="Narrator",
            content="暗い洞窟", raw_response="R1",
        )
        sqlite_store.create_scenario_turn(
            turn_id="t2", session_id="sess-1", turn_index=1,
            speaker_type="pc", speaker_id="char-haru", speaker_name="アリサ",
            content="行こう", raw_response="",
        )
        scenes = _build_trpg_scenes(
            sqlite_store, "char-haru", "はる", "太郎",
            datetime.now() - timedelta(hours=1), datetime.now() + timedelta(seconds=1),
        )
        assert len(scenes) == 1
        assert scenes[0].scene_tag == "TRPG「ダンジョン編」プレイログ"
        # 役名@キャラ名 タグ
        assert scenes[0].turns[0].speaker_tag == "Narrator"
        assert scenes[0].turns[1].speaker_tag == "アリサ@はる"

    def test_fallback_to_char_name_when_role_unresolvable(self, sqlite_store):
        """pc_assignments の slot に対応する pc_slots エントリが無いケースは `<{キャラ名}>` にフォールバック。"""
        sqlite_store.create_character("char-haru", "はる")
        sqlite_store.create_model_preset("gm", "GM", "anthropic", "claude-x")
        # pc_slots は空、でも pc_assignments には slot_id 指定がある
        # → _resolve_trpg_role_name は slot 名を引けず空文字を返す。
        scenario = _make_scenario(
            sqlite_store, title="X",
            pc_slots=[],
        )
        sqlite_store.create_scenario_session(
            session_id="sess-x", scenario_id=scenario.id, title=scenario.title,
            gm_preset_id="gm", synopsis_preset_id="gm", engine_type="ensemble_pc",
            pc_assignments=[
                {"slot_id": "ghost-slot", "player_type": "character", "character_id": "char-haru"}
            ],
        )
        sqlite_store.create_scenario_turn(
            turn_id="t1", session_id="sess-x", turn_index=0,
            speaker_type="pc", speaker_id="char-haru", speaker_name="はる",
            content="やる", raw_response="",
        )
        scenes = _build_trpg_scenes(
            sqlite_store, "char-haru", "はる", "太郎",
            datetime.now() - timedelta(hours=1), datetime.now() + timedelta(seconds=1),
        )
        assert scenes[0].turns[0].speaker_tag == "はる"

    def test_resolve_trpg_role_name(self, sqlite_store):
        """_resolve_trpg_role_name が pc_assignments → pc_slots から役名を返す。"""
        sqlite_store.create_character("char-haru", "はる")
        sqlite_store.create_model_preset("gm", "GM", "anthropic", "claude-x")
        scenario = _make_scenario(
            sqlite_store, title="X",
            pc_slots=[
                {"slot_id": "pc1", "name": "アリサ", "description": ""},
                {"slot_id": "pc2", "name": "ボブ", "description": ""},
            ],
        )
        sqlite_store.create_scenario_session(
            session_id="s1", scenario_id=scenario.id, title=scenario.title,
            gm_preset_id="gm", synopsis_preset_id="gm", engine_type="ensemble_pc",
            pc_assignments=[
                {"slot_id": "pc1", "player_type": "character", "character_id": "char-haru"},
                {"slot_id": "pc2", "player_type": "user"},
            ],
        )
        assert _resolve_trpg_role_name(sqlite_store, "s1", "char-haru") == "アリサ"


# ---------------------------------------------------------------------------
# Phase 6 — render_scenes_to_text / build_unified_pc_messages
# ---------------------------------------------------------------------------


class TestRenderScenesToText:
    """Scene 列が `<history>...</history>` 形式の正しいテキストになる。"""

    def test_basic_structure(self):
        now = datetime.now()
        scenes = [
            Scene(scene_tag="はるの日常", scene_key=("usual", 0), started_at=now, turns=[
                SpeechTurn(now, "Narrator", "朝"),
                SpeechTurn(now, "はる", "おはよう"),
            ]),
            Scene(scene_tag="太郎との対面", scene_key=("1on1", "s", 1), started_at=now, turns=[
                SpeechTurn(now, "太郎", "おかえり"),
                SpeechTurn(now, "はる", "ただいま"),
            ]),
        ]
        text = render_scenes_to_text(scenes)
        # 構造確認
        assert text.startswith("<history>")
        assert text.endswith("</history>")
        assert "<はるの日常>" in text
        assert "</はるの日常>" in text
        assert "<太郎との対面>" in text
        assert "</太郎との対面>" in text
        assert "<Narrator>朝</Narrator>" in text
        assert "<はる>おはよう</はる>" in text
        assert "<太郎>おかえり</太郎>" in text
        # 順序確認（うつつシーンが先、対面が後）
        assert text.index("<はるの日常>") < text.index("<太郎との対面>")

    def test_empty_returns_empty_history(self):
        text = render_scenes_to_text([])
        assert text == "<history>\n</history>"


class TestBuildUnifiedPcMessages:
    """end-to-end: scenario_turns + 1on1 + TRPG を時系列マージし、scene-wrap で単一メッセージ化。"""

    def test_full_timeline(self, sqlite_store):
        """前回うつつ → 1on1(text) → 1on1(face) → TRPG → 今回うつつ の流れを統合。"""
        import time
        _make_usual_session(sqlite_store)
        # 前回うつつシーン（SCENE_CLOSE で閉じる）
        sqlite_store.create_scenario_turn(
            turn_id="u1", session_id="usess", turn_index=0,
            speaker_type="narrator", speaker_id=None, speaker_name="Narrator",
            content="月曜の朝", raw_response="GM_mon_morning",
        )
        sqlite_store.create_scenario_turn(
            turn_id="u2", session_id="usess", turn_index=1,
            speaker_type="pc", speaker_id="char-haru", speaker_name="はる",
            content="出勤する", raw_response="",
        )
        sqlite_store.create_scenario_turn(
            turn_id="u3", session_id="usess", turn_index=2,
            speaker_type="narrator", speaker_id=None, speaker_name="Narrator",
            content="一日が終わった。[SCENE_CLOSE]",
            raw_response="GM_mon_close [SCENE_CLOSE]",
        )
        time.sleep(0.02)
        # 1on1 テキスト
        sqlite_store.create_chat_session("s1", "はる@p")
        _new_chat_message(sqlite_store, "s1", "user", "仕事どう？", face_to_face=0)
        _new_chat_message(sqlite_store, "s1", "character", "まあまあ", face_to_face=0, character_name="はる")
        time.sleep(0.02)
        # 1on1 対面
        _new_chat_message(sqlite_store, "s1", "user", "ただいま", face_to_face=1)
        _new_chat_message(sqlite_store, "s1", "character", "おかえり", face_to_face=1, character_name="はる")
        time.sleep(0.02)
        # TRPG
        sqlite_store.create_model_preset("gm2", "GM2", "anthropic", "claude-x")
        trpg = _make_scenario(
            sqlite_store, title="ダンジョン編",
            pc_slots=[{"slot_id": "pc1", "name": "アリサ", "description": ""}],
        )
        sqlite_store.create_scenario_session(
            session_id="trpg-1", scenario_id=trpg.id, title=trpg.title,
            gm_preset_id="gm2", synopsis_preset_id="gm2", engine_type="ensemble_pc",
            pc_assignments=[{"slot_id": "pc1", "player_type": "character", "character_id": "char-haru"}],
        )
        sqlite_store.create_scenario_turn(
            turn_id="tr1", session_id="trpg-1", turn_index=0,
            speaker_type="narrator", speaker_id=None, speaker_name="Narrator",
            content="暗い洞窟", raw_response="TR_R1",
        )
        sqlite_store.create_scenario_turn(
            turn_id="tr2", session_id="trpg-1", turn_index=1,
            speaker_type="pc", speaker_id="char-haru", speaker_name="アリサ",
            content="行こう", raw_response="",
        )
        time.sleep(0.02)
        # 今回うつつシーン（SCENE_CLOSE 後の別 raw → 次シーン）
        sqlite_store.create_scenario_turn(
            turn_id="u4", session_id="usess", turn_index=3,
            speaker_type="narrator", speaker_id=None, speaker_name="Narrator",
            content="火曜の朝", raw_response="GM_tue_morning",
        )

        history = sqlite_store.list_scenario_turns("usess")
        messages = build_unified_pc_messages(
            sqlite_store,
            history=history,
            self_character_id="char-haru",
            character_name="はる",
            self_role_name="はる",
            user_label="太郎",
        )
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        text = messages[0]["content"]

        # 全シーンタグ・主要発話が含まれる
        assert "<history>" in text and "</history>" in text
        # うつつ前回シーン
        assert "<はるの日常>" in text
        assert "<Narrator>月曜の朝</Narrator>" in text
        assert "<はる>出勤する</はる>" in text
        assert "[SCENE_CLOSE]" in text
        # 1on1 テキスト
        assert "<太郎とのテキストのやり取り>" in text
        assert "<太郎>仕事どう？</太郎>" in text
        assert "<はる>まあまあ</はる>" in text
        # 1on1 対面
        assert "<太郎との対面>" in text
        assert "<太郎>ただいま</太郎>" in text
        assert "<はる>おかえり</はる>" in text
        # TRPG（役名@キャラ名）
        assert "<TRPG「ダンジョン編」プレイログ>" in text
        assert "<アリサ@はる>行こう</アリサ@はる>" in text
        # 今回うつつシーン
        assert "<Narrator>火曜の朝</Narrator>" in text
        # うつつシーンが 2 つ（前回と今回）出現する
        assert text.count("<はるの日常>") == 2
        # 時系列順（前回うつつ → テキスト → 対面 → TRPG → 今回うつつ）
        order = [
            text.find("月曜の朝"),
            text.find("仕事どう？"),
            text.find("ただいま"),
            text.find("行こう"),
            text.find("火曜の朝"),
        ]
        assert order == sorted(order)

    def test_empty_when_nothing(self, sqlite_store):
        sqlite_store.create_character("char-haru", "はる")
        msgs = build_unified_pc_messages(
            sqlite_store,
            history=[],
            self_character_id="char-haru",
            character_name="はる",
            self_role_name="はる",
            user_label="太郎",
        )
        assert msgs == []


class TestCollectAllScenes:
    """シーン順序が started_at 昇順になる。"""

    def test_order_by_started_at(self, sqlite_store):
        """SCENE_CLOSE 後の 1on1（早い） → 新うつつシーン（遅い）の順序が保たれる。"""
        import time
        _make_usual_session(sqlite_store)
        # うつつ前回シーン（SCENE_CLOSE で閉じ、since_dt の起点になる）
        sqlite_store.create_scenario_turn(
            turn_id="u0", session_id="usess", turn_index=0,
            speaker_type="narrator", speaker_id=None, speaker_name="Narrator",
            content="閉じる。[SCENE_CLOSE]", raw_response="GM0 [SCENE_CLOSE]",
        )
        time.sleep(0.02)
        # 1on1（since_dt より後）
        sqlite_store.create_chat_session("s1", "はる@p")
        _new_chat_message(sqlite_store, "s1", "user", "ハロー")
        time.sleep(0.02)
        # 新うつつシーン（1on1 より後）
        sqlite_store.create_scenario_turn(
            turn_id="u1", session_id="usess", turn_index=1,
            speaker_type="narrator", speaker_id=None, speaker_name="Narrator",
            content="あとから来るうつつ", raw_response="GM1",
        )
        history = sqlite_store.list_scenario_turns("usess")
        scenes = collect_all_scenes(
            sqlite_store, history=history,
            self_character_id="char-haru", character_name="はる",
            user_label="太郎",
        )
        # 3 シーン: 前回うつつ → 1on1 → 新うつつ
        assert len(scenes) == 3
        assert scenes[0].started_at <= scenes[1].started_at <= scenes[2].started_at
        # シーンタグ確認
        assert scenes[0].scene_tag == "はるの日常"
        assert "太郎" in scenes[1].scene_tag
        assert scenes[2].scene_tag == "はるの日常"
