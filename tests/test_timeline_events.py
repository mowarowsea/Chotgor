"""タイムライン封筒（timeline_events）のテスト — めぐり（巡り / Aliveness）Phase 0。

検証対象（docs/aliveness_plan.md §2.1〜2.3）:
    1. dual-write: 中身テーブルへの書き込みと同時に封筒1行が追記されること
       - chat_store.create_chat_message      → chat.message
       - scenario_store.create_scenario_turn → scene.turn（参加キャラ分）
       - inscribed_memory_store.create / soft_delete → memory.inscribed / memory.forgotten
       - character_store.carve_inner_narrative → memory.carved（payload 完結型）
       - tool_event_store.add_tool_call_event → memory.recalled（power_recall のみ）
    2. retracted: 履歴の巻き戻し（編集・再生成・セッション削除）で封筒が
       削除されず retracted_at マークになること（不可逆性の担保）
    3. 読み出し: list_timeline_events のフィルタ（origin / event_type 名前空間 /
       期間 / retracted 除外 / 直近N件）
    4. バックフィル: _migrate_backfill_timeline_events が過去データから封筒を
       焼き直し、マーカーにより二重実行されないこと
"""

import uuid
from datetime import datetime, timedelta

from backend.repositories.sqlite.models import TimelineEvent


def _make_character(sqlite_store, name="はるテスト"):
    """テスト用キャラクターを1体作成して返すヘルパ。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(character_id=char_id, name=name)
    return char_id, name


def _events(sqlite_store, char_id, **kwargs):
    """封筒一覧を取得する薄いショートカット。"""
    return sqlite_store.list_timeline_events(char_id, **kwargs)


class TestChatMessageEnvelope:
    """chat.message 封筒の dual-write を検証するテストクラス。

    1on1 チャットの1発言 = 封筒1行。model_id（"{char}@{preset}"）からキャラクターを
    解決し、actor（user/character）・modality（text/face）・origin=real・
    counterpart=user が正しく焼き付くことを確認する。
    """

    def _make_session(self, sqlite_store, char_name, session_id=None):
        """キャラ名に紐づく 1on1 セッションを作るヘルパ。"""
        sid = session_id or str(uuid.uuid4())
        sqlite_store.create_chat_session(
            session_id=sid, model_id=f"{char_name}@default"
        )
        return sid

    def test_user_message_creates_envelope(self, sqlite_store):
        """ユーザ発言で actor=user・modality=text の封筒が1行できる。"""
        char_id, char_name = _make_character(sqlite_store)
        sid = self._make_session(sqlite_store, char_name)
        msg_id = str(uuid.uuid4())
        sqlite_store.create_chat_message(
            message_id=msg_id, session_id=sid, role="user", content="こんにちは"
        )
        events = _events(sqlite_store, char_id)
        assert len(events) == 1
        ev = events[0]
        assert ev.event_type == "chat.message"
        assert ev.actor == "user"
        assert ev.counterpart == "user"
        assert ev.origin == "real"
        assert ev.modality == "text"
        assert ev.session_id == sid
        assert ev.source_table == "chat_messages"
        assert ev.source_id == msg_id
        assert ev.retracted_at is None

    def test_character_message_face_to_face(self, sqlite_store):
        """キャラ発言（対面モード）で actor=character・modality=face になる。"""
        char_id, char_name = _make_character(sqlite_store)
        sid = self._make_session(sqlite_store, char_name)
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()), session_id=sid,
            role="character", content="やあ", character_name=char_name,
            face_to_face=1,
        )
        ev = _events(sqlite_store, char_id)[0]
        assert ev.actor == "character"
        assert ev.modality == "face"

    def test_system_message_skips_envelope(self, sqlite_store):
        """退席通知などのシステムメッセージは封筒に載らない。"""
        char_id, char_name = _make_character(sqlite_store)
        sid = self._make_session(sqlite_store, char_name)
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()), session_id=sid,
            role="character", content="（退席しました）",
            is_system_message=True,
        )
        assert _events(sqlite_store, char_id) == []

    def test_unknown_character_skips_envelope(self, sqlite_store):
        """model_id のキャラ名が解決できないセッションでは封筒を作らない（メッセージは保存される）。"""
        sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(
            session_id=sid, model_id="存在しないキャラ@default"
        )
        msg = sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()), session_id=sid,
            role="user", content="やっほー",
        )
        assert msg is not None
        with sqlite_store.get_session() as s:
            assert s.query(TimelineEvent).count() == 0

    def test_delete_from_marks_retracted(self, sqlite_store):
        """編集・再生成の巻き戻し（delete_chat_messages_from）で封筒が retracted になる。

        封筒は物理削除されず、既定の読み出しから消え、include_retracted=True でだけ見える。
        巻き戻し境界より前の封筒は無傷であること。
        """
        char_id, char_name = _make_character(sqlite_store)
        sid = self._make_session(sqlite_store, char_name)
        ids = []
        for i in range(3):
            mid = str(uuid.uuid4())
            ids.append(mid)
            sqlite_store.create_chat_message(
                message_id=mid, session_id=sid, role="user", content=f"m{i}"
            )
        # 2件目以降を巻き戻し
        assert sqlite_store.delete_chat_messages_from(sid, ids[1]) is True
        visible = _events(sqlite_store, char_id)
        assert [e.source_id for e in visible] == [ids[0]]
        all_events = _events(sqlite_store, char_id, include_retracted=True)
        assert len(all_events) == 3
        retracted = [e for e in all_events if e.retracted_at is not None]
        assert sorted(e.source_id for e in retracted) == sorted(ids[1:])

    def test_delete_session_marks_retracted(self, sqlite_store):
        """セッションごと削除しても封筒は retracted マークで存在記録が残る。"""
        char_id, char_name = _make_character(sqlite_store)
        sid = self._make_session(sqlite_store, char_name)
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()), session_id=sid, role="user", content="a"
        )
        sqlite_store.delete_chat_session(sid)
        assert _events(sqlite_store, char_id) == []
        assert len(_events(sqlite_store, char_id, include_retracted=True)) == 1


class TestSceneTurnEnvelope:
    """scene.turn 封筒の dual-write を検証するテストクラス。

    うつつ（usual_days）では世界所有者キャラ1人・origin=usual、
    通常シナリオでは pc_assignments の character 配役全員・origin=interlude の
    封筒ができる。actor はタイムライン所有者から見た話者
    （自分=character / 他キャラPC・NPC=npc:<名前> / user / narrator）。
    """

    def _make_usual_world(self, sqlite_store, owner_id):
        """うつつ（usual_days）シナリオ＋セッションを作るヘルパ。"""
        scen_id = str(uuid.uuid4())
        sqlite_store.create_scenario(
            scenario_id=scen_id, title="うつつ世界",
            owner_character_id=owner_id,
        )
        sess_id = str(uuid.uuid4())
        sqlite_store.create_scenario_session(
            session_id=sess_id, scenario_id=scen_id, title="うつつ",
            gm_preset_id="p1", synopsis_preset_id="p1",
            engine_type="usual_days",
        )
        return sess_id

    def test_usual_turn_creates_owner_envelope(self, sqlite_store):
        """うつつのターンは所有者キャラの封筒（origin=usual）になる。"""
        char_id, char_name = _make_character(sqlite_store)
        sess_id = self._make_usual_world(sqlite_store, char_id)
        turn_id = str(uuid.uuid4())
        sqlite_store.create_scenario_turn(
            turn_id=turn_id, session_id=sess_id, turn_index=0,
            speaker_type="character", speaker_name=char_name,
            speaker_id=char_id, content="散歩に出た",
        )
        events = _events(sqlite_store, char_id)
        assert len(events) == 1
        ev = events[0]
        assert ev.event_type == "scene.turn"
        assert ev.actor == "character"
        assert ev.origin == "usual"
        assert ev.source_table == "scenario_turns"
        assert ev.source_id == turn_id

    def test_usual_npc_and_narrator_actor(self, sqlite_store):
        """NPC 発話は actor=npc:<名前>、Narrator は narrator になる。"""
        char_id, _ = _make_character(sqlite_store)
        sess_id = self._make_usual_world(sqlite_store, char_id)
        sqlite_store.create_scenario_turn(
            turn_id=str(uuid.uuid4()), session_id=sess_id, turn_index=0,
            speaker_type="npc", speaker_name="店主", content="いらっしゃい",
        )
        sqlite_store.create_scenario_turn(
            turn_id=str(uuid.uuid4()), session_id=sess_id, turn_index=1,
            speaker_type="narrator", speaker_name="Narrator", content="夕暮れ",
        )
        actors = [e.actor for e in _events(sqlite_store, char_id)]
        assert actors == ["npc:店主", "narrator"]

    def test_ensemble_pc_turn_creates_envelope_per_character(self, sqlite_store):
        """通常シナリオでは character 配役全員に封筒ができ、actor は観測者相対になる。

        キャラA本人の発話は、Aのタイムラインでは actor=character、
        Bのタイムラインでは actor=npc:<Aの名前> として記録される。
        """
        a_id, a_name = _make_character(sqlite_store, name="キャラA")
        b_id, b_name = _make_character(sqlite_store, name="キャラB")
        scen_id = str(uuid.uuid4())
        sqlite_store.create_scenario(scenario_id=scen_id, title="冒険")
        sess_id = str(uuid.uuid4())
        sqlite_store.create_scenario_session(
            session_id=sess_id, scenario_id=scen_id, title="冒険",
            gm_preset_id="p1", synopsis_preset_id="p1",
            engine_type="ensemble_pc",
            pc_assignments=[
                {"slot_id": "pc1", "player_type": "character", "character_id": a_id},
                {"slot_id": "pc2", "player_type": "character", "character_id": b_id},
                {"slot_id": "user", "player_type": "user"},
            ],
        )
        sqlite_store.create_scenario_turn(
            turn_id=str(uuid.uuid4()), session_id=sess_id, turn_index=0,
            speaker_type="character", speaker_name=a_name,
            speaker_id=a_id, content="行くぞ",
        )
        ev_a = _events(sqlite_store, a_id)
        ev_b = _events(sqlite_store, b_id)
        assert len(ev_a) == 1 and len(ev_b) == 1
        assert ev_a[0].actor == "character"
        assert ev_b[0].actor == f"npc:{a_name}"
        assert ev_a[0].origin == "interlude"

    def test_no_character_participants_no_envelope(self, sqlite_store):
        """character 配役のいないシナリオ（ユーザとNPCだけ）は封筒を作らない。"""
        scen_id = str(uuid.uuid4())
        sqlite_store.create_scenario(scenario_id=scen_id, title="独り旅")
        sess_id = str(uuid.uuid4())
        sqlite_store.create_scenario_session(
            session_id=sess_id, scenario_id=scen_id, title="独り旅",
            gm_preset_id="p1", synopsis_preset_id="p1",
        )
        sqlite_store.create_scenario_turn(
            turn_id=str(uuid.uuid4()), session_id=sess_id, turn_index=0,
            speaker_type="narrator", speaker_name="Narrator", content="朝",
        )
        with sqlite_store.get_session() as s:
            assert s.query(TimelineEvent).count() == 0

    def test_delete_turns_from_marks_retracted(self, sqlite_store):
        """ターンの巻き戻し（delete_scenario_turns_from）で封筒が retracted になる。"""
        char_id, char_name = _make_character(sqlite_store)
        sess_id = self._make_usual_world(sqlite_store, char_id)
        turn_ids = []
        for i in range(3):
            tid = str(uuid.uuid4())
            turn_ids.append(tid)
            sqlite_store.create_scenario_turn(
                turn_id=tid, session_id=sess_id, turn_index=i,
                speaker_type="narrator", speaker_name="Narrator", content=f"t{i}",
            )
        assert sqlite_store.delete_scenario_turns_from(sess_id, turn_ids[1]) is True
        visible = _events(sqlite_store, char_id)
        assert [e.source_id for e in visible] == [turn_ids[0]]
        assert len(_events(sqlite_store, char_id, include_retracted=True)) == 3

    def test_delete_session_marks_retracted(self, sqlite_store):
        """シナリオセッション削除で全ターンの封筒が retracted になる。"""
        char_id, char_name = _make_character(sqlite_store)
        sess_id = self._make_usual_world(sqlite_store, char_id)
        sqlite_store.create_scenario_turn(
            turn_id=str(uuid.uuid4()), session_id=sess_id, turn_index=0,
            speaker_type="narrator", speaker_name="Narrator", content="a",
        )
        sqlite_store.delete_scenario_session(sess_id)
        assert _events(sqlite_store, char_id) == []
        assert len(_events(sqlite_store, char_id, include_retracted=True)) == 1


class TestMemoryEnvelopes:
    """memory.inscribed / memory.forgotten / memory.carved 封筒を検証するテストクラス。

    記憶の刻み込み・忘却・inner_narrative の彫り込みが、いずれも actor=character の
    封筒として正本に残ること、復元（restore）が memory.forgotten 封筒だけを
    retract すること（記憶を刻んだ事実は消えない）を確認する。
    """

    def test_inscribe_creates_envelope(self, sqlite_store):
        """記憶の刻み込みで memory.inscribed 封筒（origin 継承）ができる。"""
        char_id, _ = _make_character(sqlite_store)
        mem_id = str(uuid.uuid4())
        sqlite_store.create_inscribed_memory(
            memory_id=mem_id, character_id=char_id,
            content="今日の散歩は気持ちよかった", origin="usual",
        )
        ev = _events(sqlite_store, char_id)[0]
        assert ev.event_type == "memory.inscribed"
        assert ev.actor == "character"
        assert ev.origin == "usual"
        assert ev.source_table == "inscribed_memories"
        assert ev.source_id == mem_id

    def test_soft_delete_creates_forgotten_envelope(self, sqlite_store):
        """忘却（soft delete）で memory.forgotten 封筒が追加される。"""
        char_id, _ = _make_character(sqlite_store)
        mem_id = str(uuid.uuid4())
        sqlite_store.create_inscribed_memory(
            memory_id=mem_id, character_id=char_id, content="些細なこと",
        )
        assert sqlite_store.soft_delete_inscribed_memory(mem_id) is True
        types = [e.event_type for e in _events(sqlite_store, char_id)]
        assert types == ["memory.inscribed", "memory.forgotten"]

    def test_restore_retracts_only_forgotten(self, sqlite_store):
        """復元で memory.forgotten だけ retracted になり、memory.inscribed は残る。"""
        char_id, _ = _make_character(sqlite_store)
        mem_id = str(uuid.uuid4())
        sqlite_store.create_inscribed_memory(
            memory_id=mem_id, character_id=char_id, content="消して戻す",
        )
        sqlite_store.soft_delete_inscribed_memory(mem_id)
        assert sqlite_store.restore_inscribed_memory(mem_id) is True
        visible = _events(sqlite_store, char_id)
        assert [e.event_type for e in visible] == ["memory.inscribed"]
        all_ev = _events(sqlite_store, char_id, include_retracted=True)
        forgotten = [e for e in all_ev if e.event_type == "memory.forgotten"]
        assert len(forgotten) == 1 and forgotten[0].retracted_at is not None

    def test_carve_creates_payload_envelope(self, sqlite_store):
        """carve_inner_narrative が inner_narrative 更新＋memory.carved 封筒を同時に行う。

        payload 完結型（source_table=None）で、彫った内容と mode が payload に残ること、
        append が改行区切りで追記されることを確認する。
        """
        char_id, _ = _make_character(sqlite_store)
        sqlite_store.carve_inner_narrative(char_id, "append", "わたしは歌が好き")
        char = sqlite_store.get_character(char_id)
        assert char.inner_narrative == "わたしは歌が好き"
        sqlite_store.carve_inner_narrative(char_id, "append", "朝は苦手")
        char = sqlite_store.get_character(char_id)
        assert char.inner_narrative == "わたしは歌が好き\n朝は苦手"
        events = _events(sqlite_store, char_id)
        assert [e.event_type for e in events] == ["memory.carved", "memory.carved"]
        ev = events[0]
        assert ev.actor == "character"
        assert ev.source_table is None
        assert ev.payload == {"mode": "append", "content": "わたしは歌が好き"}

    def test_carve_overwrite(self, sqlite_store):
        """overwrite モードは inner_narrative を全置換し payload.mode=overwrite で残る。"""
        char_id, _ = _make_character(sqlite_store)
        sqlite_store.carve_inner_narrative(char_id, "append", "古い物語")
        sqlite_store.carve_inner_narrative(char_id, "overwrite", "新しい物語")
        char = sqlite_store.get_character(char_id)
        assert char.inner_narrative == "新しい物語"
        last = _events(sqlite_store, char_id)[-1]
        assert last.payload["mode"] == "overwrite"


class TestRecalledEnvelope:
    """memory.recalled 封筒（power_recall のみ）を検証するテストクラス。

    ツール実行イベントの記録経路（add_tool_call_event）のうち、能動的想起
    （power_recall・status=ok）だけが封筒に載り、他のツールや失敗した実行、
    キャラ解決不能な実行は載らないことを確認する。
    """

    def test_power_recall_creates_envelope(self, sqlite_store):
        """power_recall（ok）で memory.recalled 封筒ができ、source が tool_call_events を指す。"""
        char_id, char_name = _make_character(sqlite_store)
        sqlite_store.add_tool_call_event(
            tool_name="power_recall",
            arguments_json='{"query": "あの約束"}',
            target=char_name, feature="chat",
        )
        ev = _events(sqlite_store, char_id)[0]
        assert ev.event_type == "memory.recalled"
        assert ev.actor == "character"
        assert ev.origin == "real"
        assert ev.source_table == "tool_call_events"
        assert ev.source_id  # autoincrement id が入っている

    def test_usual_feature_maps_origin(self, sqlite_store):
        """うつつ由来（feature=usual_days_pc）の想起は origin=usual になる。"""
        char_id, char_name = _make_character(sqlite_store)
        sqlite_store.add_tool_call_event(
            tool_name="power_recall", target=char_name, feature="usual_days_pc",
        )
        assert _events(sqlite_store, char_id)[0].origin == "usual"

    def test_other_tools_and_errors_skip_envelope(self, sqlite_store):
        """power_recall 以外のツール・エラー実行・未知キャラは封筒に載らない。"""
        char_id, char_name = _make_character(sqlite_store)
        sqlite_store.add_tool_call_event(
            tool_name="inscribe_memory", target=char_name, feature="chat",
        )
        sqlite_store.add_tool_call_event(
            tool_name="power_recall", target=char_name, feature="chat",
            status="error", error_message="失敗",
        )
        sqlite_store.add_tool_call_event(
            tool_name="power_recall", target="いないキャラ", feature="chat",
        )
        assert _events(sqlite_store, char_id) == []


class TestTimelineQuery:
    """list_timeline_events の各フィルタと record_timeline_event を検証するテストクラス。

    origin フィルタ・event_type 名前空間（前方一致）フィルタ・期間フィルタ・
    newest_first + limit（直近N件）・payload 完結型イベントの直書きを確認する。
    """

    def test_record_and_filters(self, sqlite_store):
        """直書きイベントの各フィルタが期待通り効く。"""
        char_id, _ = _make_character(sqlite_store)
        base = datetime(2026, 7, 1, 12, 0, 0)
        specs = [
            ("chat.message", "real", base),
            ("chat.farewell", "real", base + timedelta(hours=1)),
            ("scene.turn", "usual", base + timedelta(hours=2)),
            ("night.chronicle", "real", base + timedelta(hours=3)),
        ]
        for ev_type, origin, at in specs:
            sqlite_store.record_timeline_event(
                character_id=char_id, event_type=ev_type,
                origin=origin, occurred_at=at,
            )
        # 名前空間フィルタ（"chat." は message と farewell の両方に当たる）
        chats = _events(sqlite_store, char_id, event_type_prefixes=["chat."])
        assert [e.event_type for e in chats] == ["chat.message", "chat.farewell"]
        # origin フィルタ
        usual = _events(sqlite_store, char_id, origins=["usual"])
        assert [e.event_type for e in usual] == ["scene.turn"]
        # 期間フィルタ（until は exclusive）
        mid = _events(
            sqlite_store, char_id,
            since=base + timedelta(hours=1), until=base + timedelta(hours=3),
        )
        assert [e.event_type for e in mid] == ["chat.farewell", "scene.turn"]
        # 直近N件（newest_first + limit）
        latest = _events(sqlite_store, char_id, newest_first=True, limit=2)
        assert [e.event_type for e in latest] == ["night.chronicle", "scene.turn"]

    def test_count_by_source(self, sqlite_store):
        """count_timeline_events_by_source は retracted も含めた件数を返す（計器の突合用）。"""
        char_id, char_name = _make_character(sqlite_store)
        sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=sid, model_id=f"{char_name}@d")
        ids = []
        for i in range(2):
            mid = str(uuid.uuid4())
            ids.append(mid)
            sqlite_store.create_chat_message(
                message_id=mid, session_id=sid, role="user", content=f"m{i}"
            )
        sqlite_store.delete_chat_messages_from(sid, ids[1])
        assert sqlite_store.count_timeline_events_by_source("chat_messages") == 2


class TestBackfillMigration:
    """バックフィルマイグレーションを検証するテストクラス。

    手順: dual-write で作られた封筒とマーカーをいったん全消しして「旧DB」状態を
    再現し、_migrate_backfill_timeline_events() を直接呼ぶ。過去データから封筒が
    焼き直されること・マーカーによる冪等性（二重実行で増えない）ことを確認する。
    """

    def _wipe_envelopes(self, sqlite_store):
        """封筒とバックフィルマーカーを全削除して「導入前の旧DB」を再現する。"""
        with sqlite_store.engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM timeline_events")
            conn.exec_driver_sql(
                "DELETE FROM global_settings WHERE key='timeline_backfill_done'"
            )

    def test_backfill_recreates_envelopes(self, sqlite_store):
        """chat / scene / memory の過去行から封筒が焼き直される。"""
        char_id, char_name = _make_character(sqlite_store)
        # 1on1 メッセージ2件（うちシステム1件は対象外）
        sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=sid, model_id=f"{char_name}@d")
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()), session_id=sid, role="user", content="hi"
        )
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()), session_id=sid, role="character",
            content="sys", is_system_message=True,
        )
        # うつつターン1件
        scen_id = str(uuid.uuid4())
        sqlite_store.create_scenario(
            scenario_id=scen_id, title="世界", owner_character_id=char_id
        )
        usess_id = str(uuid.uuid4())
        sqlite_store.create_scenario_session(
            session_id=usess_id, scenario_id=scen_id, title="うつつ",
            gm_preset_id="p", synopsis_preset_id="p", engine_type="usual_days",
        )
        sqlite_store.create_scenario_turn(
            turn_id=str(uuid.uuid4()), session_id=usess_id, turn_index=0,
            speaker_type="npc", speaker_name="店主", content="どうも",
        )
        # 記憶（1件は忘却済み）
        m1, m2 = str(uuid.uuid4()), str(uuid.uuid4())
        sqlite_store.create_inscribed_memory(
            memory_id=m1, character_id=char_id, content="残る記憶"
        )
        sqlite_store.create_inscribed_memory(
            memory_id=m2, character_id=char_id, content="消える記憶"
        )
        sqlite_store.soft_delete_inscribed_memory(m2)
        # power_recall のツールイベント
        sqlite_store.add_tool_call_event(
            tool_name="power_recall", arguments_json='{"query":"q"}',
            target=char_name, feature="chat",
        )

        self._wipe_envelopes(sqlite_store)
        sqlite_store._migrate_backfill_timeline_events()

        types = sorted(e.event_type for e in _events(sqlite_store, char_id))
        assert types == sorted([
            "chat.message",        # ユーザ発言のみ（システムメッセージ除外）
            "scene.turn",
            "memory.inscribed", "memory.inscribed", "memory.forgotten",
            "memory.recalled",
        ])

    def test_backfill_idempotent(self, sqlite_store):
        """マーカーにより二重実行しても封筒が増えない。"""
        char_id, char_name = _make_character(sqlite_store)
        sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=sid, model_id=f"{char_name}@d")
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()), session_id=sid, role="user", content="a"
        )
        self._wipe_envelopes(sqlite_store)
        sqlite_store._migrate_backfill_timeline_events()
        count1 = len(_events(sqlite_store, char_id))
        sqlite_store._migrate_backfill_timeline_events()
        count2 = len(_events(sqlite_store, char_id))
        assert count1 == count2 == 1
