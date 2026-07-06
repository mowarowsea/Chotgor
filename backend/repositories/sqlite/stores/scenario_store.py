"""Scenario Chat の CRUD — SQLiteStore Mixin。

`scenario_chat` 機能で使う以下 4 テーブルへの永続化を担う:
  - scenarios         : シナリオテンプレート（タイトル・GM・世界観・NPCの定義先）
  - scenario_npcs     : テンプレートに紐づく NPC（軽量 Character）
  - scenario_sessions : テンプレから起動されたプレイインスタンス
  - scenario_turns    : 発話履歴（session_id に紐づく）

設計方針:
  - シナリオは何度でも遊べる「テンプレート」。
  - セッションはプレイ毎に起動される薄いインスタンスで、テンプレを参照する。
  - シナリオを「編集」してもプレイ中のセッションは影響を受けず、
    シナリオを「削除」するときは紐づくセッション・ターンも一括削除する。
  - NPC はシナリオ側で管理する。プレイ画面では追加・編集しない。
"""

from datetime import datetime

class ScenarioChatStoreMixin:
    """シナリオテンプレ・セッション・NPC・ターンの作成・取得・更新・削除を担う Mixin。"""

    # ────────────────────────────────────────────────────────────────────────
    #   Scenario Templates
    # ────────────────────────────────────────────────────────────────────────

    def create_scenario(
        self,
        scenario_id: str,
        title: str,
        scenario: str | None = None,
        intro: str | None = None,
        history_max_turns: int | None = None,
        history_max_chars: int | None = None,
        custom_system_prompt: str | None = None,
        dice_pool_spec: dict | None = None,
        pc_slots: list[dict] | None = None,
        banner_data: str | None = None,
        owner_character_id: str | None = None,
        usual_config: dict | None = None,
    ):
        """シナリオテンプレートを新規作成する。

        場所・空気・語り口・テンポなどはすべて `scenario` テキストにまとめて記述する。
        intro はセッション開始時に固定ターンとして挿入される導入部（@キャラ: 記法）。
        custom_system_prompt はGMシステムプロンプトの完全カスタマイズ。
                         空の場合、デフォルトテンプレートが自動設定される。
        dice_pool_spec は ensemble_pc エンジン時に毎ターン乱数生成する種別と本数の dict。
                         例: {"d6": 10, "d100": 5}。NULL なら engine 側既定値 {"d6": 10}。
        pc_slots は ensemble_pc エンジン時の PC枠定義。
                         [{"slot_id":"pc1","name":"アリス","description":"剣士。商家の出。...","image_data":"data:image/..."}]。
                         シナリオ側で人物像・知っていることを含めて記述する。
                         セッション開始時に各枠を「ユーザが演じる/AIキャラが演じる」と割り振る。
        banner_data はシナリオのバナー画像（base64 data URI）。一覧・編集画面の見栄え用。
        owner_character_id はうつつ（Usual Days）所有者キャラ ID。値ありなら、そのキャラの
                         「生活世界」専用シナリオとして汎用一覧から除外される。NULL=汎用シナリオ。
        usual_config はうつつ運用設定（有効化トグル・スロット時刻・時間グリッド等）の dict。
                         owner_character_id が NULL のときは未使用。

        旧 user_alias は廃止。ユーザPCも pc_slots の 1 枠として定義する
        （セッション開始時に player_type="user" を割り当てる）。

        GM の LLM プリセットはテンプレートには持たない（セッション単位で選択する）。
        """
        # custom_system_prompt が None または空の場合、デフォルトテンプレートを設定
        if not custom_system_prompt:
            from backend.services.scenario_chat.prompt_builder import DEFAULT_GM_SYSTEM_PROMPT_TEMPLATE
            custom_system_prompt = DEFAULT_GM_SYSTEM_PROMPT_TEMPLATE

        with self.get_session() as session:
            from backend.repositories.sqlite.store import Scenario
            obj = Scenario(
                id=scenario_id,
                title=title,
                scenario=scenario,
                intro=intro,
                history_max_turns=history_max_turns,
                history_max_chars=history_max_chars,
                custom_system_prompt=custom_system_prompt,
                dice_pool_spec=dice_pool_spec,
                pc_slots=pc_slots,
                banner_data=banner_data,
                owner_character_id=owner_character_id,
                usual_config=usual_config,
            )
            session.add(obj)
            session.commit()
            session.refresh(obj)
            return obj

    def get_scenario(self, scenario_id: str):
        """ID でシナリオテンプレートを取得する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import Scenario
            return session.get(Scenario, scenario_id)

    def list_scenarios(self, limit: int = 100, include_usual: bool = False) -> list:
        """シナリオテンプレート一覧を更新日時の新しい順で返す。

        既定では汎用シナリオのみ（``owner_character_id IS NULL``）を返し、
        うつつ（Usual Days）専用シナリオは除外する。うつつ世界は「キャラ固有の生活世界」で
        汎用シナリオ一覧には並べない方針（1 キャラ 1 世界）。
        ``include_usual=True`` のときだけ、うつつ専用シナリオも含めて返す。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import Scenario
            query = session.query(Scenario)
            if not include_usual:
                query = query.filter(Scenario.owner_character_id.is_(None))
            return (
                query
                .order_by(Scenario.updated_at.desc())
                .limit(limit)
                .all()
            )

    def get_usual_scenario(self, owner_character_id: str):
        """指定キャラのうつつ（生活世界）シナリオを返す（無ければ None）。

        1 キャラ 1 世界の前提で、``owner_character_id`` 一致の最初の 1 件を返す。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import Scenario
            return (
                session.query(Scenario)
                .filter(Scenario.owner_character_id == owner_character_id)
                .order_by(Scenario.updated_at.desc())
                .first()
            )

    def list_usual_scenarios(self) -> list:
        """全うつつ（生活世界）シナリオを返す（``owner_character_id IS NOT NULL``）。

        スケジューラ（_usual_days_scheduler）が有効なうつつ世界を走査するために使う。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import Scenario
            return (
                session.query(Scenario)
                .filter(Scenario.owner_character_id.isnot(None))
                .order_by(Scenario.updated_at.desc())
                .all()
            )

    def update_scenario(self, scenario_id: str, **kwargs):
        """シナリオテンプレートを部分更新する。

        プレイ中のセッションは影響を受けない（セッションはテンプレを実行時 lookup するため、
        次の発話から新しいテンプレ内容が反映される）。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import Scenario
            obj = session.get(Scenario, scenario_id)
            if not obj:
                return None
            for k, v in kwargs.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
            obj.updated_at = datetime.now()
            session.commit()
            session.refresh(obj)
            return obj

    def delete_scenario(self, scenario_id: str) -> bool:
        """シナリオテンプレートを削除する。紐づく NPC・セッション・ターンも全て削除する。

        Returns:
            削除成功なら True、テンプレートが存在しなければ False。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import (
                Scenario,
                ScenarioNpc,
                ScenarioSession,
                ScenarioTurn,
            )
            # 紐づくセッションのターンを一括削除
            session_ids = [
                row[0]
                for row in session.query(ScenarioSession.id)
                .filter(ScenarioSession.scenario_id == scenario_id)
                .all()
            ]
            if session_ids:
                session.query(ScenarioTurn).filter(
                    ScenarioTurn.session_id.in_(session_ids)
                ).delete(synchronize_session=False)
                session.query(ScenarioSession).filter(
                    ScenarioSession.scenario_id == scenario_id
                ).delete(synchronize_session=False)
            # NPC を削除
            session.query(ScenarioNpc).filter(
                ScenarioNpc.scenario_id == scenario_id
            ).delete(synchronize_session=False)
            obj = session.get(Scenario, scenario_id)
            if not obj:
                session.commit()
                return False
            session.delete(obj)
            session.commit()
            return True

    # ────────────────────────────────────────────────────────────────────────
    #   Scenario NPCs（テンプレートに紐づく）
    # ────────────────────────────────────────────────────────────────────────

    def create_scenario_npc(
        self,
        npc_id: str,
        scenario_id: str,
        name: str,
        description: str | None = None,
        image_data: str | None = None,
    ):
        """シナリオテンプレート内 NPC を作成する。

        Args:
            description: 人物像・口調・話し方を自由テキストで記述。
            image_data: アバター画像（base64 data URI 形式）。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioNpc
            obj = ScenarioNpc(
                id=npc_id,
                scenario_id=scenario_id,
                name=name,
                description=description,
                image_data=image_data,
            )
            session.add(obj)
            session.commit()
            session.refresh(obj)
            return obj

    def get_scenario_npc(self, npc_id: str):
        """ID で NPC を取得する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioNpc
            return session.get(ScenarioNpc, npc_id)

    def list_scenario_npcs(self, scenario_id: str) -> list:
        """シナリオテンプレートに紐づく NPC を作成順で返す。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioNpc
            return (
                session.query(ScenarioNpc)
                .filter(ScenarioNpc.scenario_id == scenario_id)
                .order_by(ScenarioNpc.created_at.asc())
                .all()
            )

    def update_scenario_npc(self, npc_id: str, **kwargs):
        """NPC のフィールドを更新する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioNpc
            obj = session.get(ScenarioNpc, npc_id)
            if not obj:
                return None
            for k, v in kwargs.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
            session.commit()
            session.refresh(obj)
            return obj

    def delete_scenario_npc(self, npc_id: str) -> bool:
        """NPC を削除する。過去の発話履歴（scenario_turns.speaker_id）は残るが参照不能になる。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioNpc
            obj = session.get(ScenarioNpc, npc_id)
            if not obj:
                return False
            session.delete(obj)
            session.commit()
            return True

    # ────────────────────────────────────────────────────────────────────────
    #   Scenario Sessions（プレイインスタンス）
    # ────────────────────────────────────────────────────────────────────────

    def create_scenario_session(
        self,
        session_id: str,
        scenario_id: str,
        title: str,
        gm_preset_id: str,
        synopsis_preset_id: str,
        engine_type: str = "ensemble",
        pc_assignments: list[dict] | None = None,
    ):
        """シナリオから新しいプレイセッションを起動する。

        Args:
            session_id: セッション UUID。
            scenario_id: 紐づくシナリオテンプレート ID。
            title: 起動時のセッションタイトル（通常はテンプレ title をコピー）。
            gm_preset_id: このセッションで使う GM の LLM プリセット ID。
                          後から `update_scenario_session(gm_preset_id=...)` で変更可。
            synopsis_preset_id: あらすじ蒸留専用の LLM プリセット ID。
                                レートリミット節約用に GM とは別モデルを選べる。
                                通常 GM と同じプリセットを指定すれば従来挙動と同じ。
            engine_type: "ensemble"（GMのみ）または "ensemble_pc"（GM+PC配役）。
            pc_assignments: ensemble_pc 専用。スロット割当てリスト。
                            [{"slot_id":"pc1","player_type":"user"|"character",
                              "character_id":"...","preset_id":"..."}]。
                            player_type="user" はそのスロットをユーザが演じる（character_id/preset_id 不要）。
                            player_type="character" は Chotgor キャラが演じる（character_id 必須）。
                            ensemble エンジン時は NULL/省略。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioSession
            obj = ScenarioSession(
                id=session_id,
                scenario_id=scenario_id,
                title=title,
                gm_preset_id=gm_preset_id,
                synopsis_preset_id=synopsis_preset_id,
                engine_type=engine_type,
                status="active",
                pc_assignments=pc_assignments,
            )
            session.add(obj)
            session.commit()
            session.refresh(obj)
            return obj

    def get_scenario_session(self, session_id: str):
        """ID でプレイセッションを取得する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioSession
            return session.get(ScenarioSession, session_id)

    def list_scenario_sessions(self, limit: int = 100, include_usual: bool = False) -> list:
        """プレイセッション一覧を更新日時の新しい順で返す。

        既定では うつつ（engine_type="usual_days"）の無人セッションを除外する。
        うつつは「キャラ固有の生活世界」で通常のシナリオ一覧には並べない方針
        （生ログは /ui/logs でのみ覗ける）。``include_usual=True`` で含める。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioSession
            query = session.query(ScenarioSession)
            if not include_usual:
                query = query.filter(ScenarioSession.engine_type != "usual_days")
            return (
                query
                .order_by(ScenarioSession.updated_at.desc())
                .limit(limit)
                .all()
            )

    def update_scenario_turn(self, turn_id: str, **kwargs):
        """発話ターンのフィールド（content / raw_response 等）を部分更新する。

        うつつの [SCENE_CLOSE] マーカーを表示用 content から取り除くなど、保存後の
        ターン本文を整える用途で使う。存在しなければ None を返す。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioTurn
            obj = session.get(ScenarioTurn, turn_id)
            if not obj:
                return None
            for k, v in kwargs.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
            session.commit()
            session.refresh(obj)
            return obj

    def list_scenario_sessions_by_scenario(self, scenario_id: str) -> list:
        """指定シナリオから起動された全プレイセッションを返す。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioSession
            return (
                session.query(ScenarioSession)
                .filter(ScenarioSession.scenario_id == scenario_id)
                .order_by(ScenarioSession.updated_at.desc())
                .all()
            )

    def get_scenario_session_titles_by_ids(
        self, session_ids: list[str]
    ) -> dict[str, str]:
        """複数の session_id に対応するタイトルを IN 句で一括取得して返す。

        ログ一覧の見出し統一（セッション名+プリセット表示）で利用する。
        該当が無い ID はキー自体を返さない。

        Args:
            session_ids: scenario_sessions.id のリスト。

        Returns:
            {session_id: title} の辞書。
        """
        if not session_ids:
            return {}
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioSession
            rows = (
                session.query(ScenarioSession.id, ScenarioSession.title)
                .filter(ScenarioSession.id.in_(session_ids))
                .all()
            )
            return {sid: title for sid, title in rows}

    def update_scenario_session(self, session_id: str, **kwargs):
        """プレイセッションのフィールド（title / status 等）を更新する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioSession
            obj = session.get(ScenarioSession, session_id)
            if not obj:
                return None
            for k, v in kwargs.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
            obj.updated_at = datetime.now()
            session.commit()
            session.refresh(obj)
            return obj

    def get_scenario_session_synopsis(self, session_id: str) -> dict | None:
        """セッションのあらすじ（auto / manual / last_turn_index）を取得する。

        Returns:
            セッションが存在すれば dict、存在しなければ None。
            dict は {"auto": str, "manual": str, "last_turn_index": int} の形。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioSession
            obj = session.get(ScenarioSession, session_id)
            if not obj:
                return None
            return {
                "auto": obj.synopsis_auto or "",
                "manual": obj.synopsis_manual or "",
                "last_turn_index": int(obj.synopsis_last_turn_index)
                if obj.synopsis_last_turn_index is not None
                else -1,
            }

    def update_scenario_session_synopsis(
        self,
        session_id: str,
        *,
        auto: str | None = None,
        manual: str | None = None,
        last_turn_index: int | None = None,
    ) -> dict | None:
        """セッションのあらすじを部分更新する。

        引数で None を渡したフィールドは触らない。auto を None にすれば
        ユーザが手編集した auto を保護できる（自動追記フローでは last_turn_index 更新と同時に
        auto に新規要約を「追記済みの結果」として渡すこと）。

        Returns:
            更新後のあらすじ dict（get と同じ形）。セッション未存在は None。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioSession
            obj = session.get(ScenarioSession, session_id)
            if not obj:
                return None
            if auto is not None:
                obj.synopsis_auto = auto
            if manual is not None:
                obj.synopsis_manual = manual
            if last_turn_index is not None:
                obj.synopsis_last_turn_index = int(last_turn_index)
            obj.updated_at = datetime.now()
            session.commit()
            session.refresh(obj)
            return {
                "auto": obj.synopsis_auto or "",
                "manual": obj.synopsis_manual or "",
                "last_turn_index": int(obj.synopsis_last_turn_index)
                if obj.synopsis_last_turn_index is not None
                else -1,
            }

    def delete_scenario_session(self, session_id: str) -> bool:
        """プレイセッションを削除する。ターンも一括削除する。

        テンプレ（scenarios）には影響しない。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioSession, ScenarioTurn
            # 封筒は削除せず retracted マーク（セッションごと消しても存在記録は残す）
            turn_ids = [
                row[0]
                for row in session.query(ScenarioTurn.id)
                .filter(ScenarioTurn.session_id == session_id)
                .all()
            ]
            self._retract_timeline_events_in_session(
                session, "scenario_turns", turn_ids
            )
            session.query(ScenarioTurn).filter(ScenarioTurn.session_id == session_id).delete()
            obj = session.get(ScenarioSession, session_id)
            if not obj:
                session.commit()
                return False
            session.delete(obj)
            session.commit()
            return True

    # ────────────────────────────────────────────────────────────────────────
    #   Scenario Turns（プレイインスタンスに紐づく発話履歴）
    # ────────────────────────────────────────────────────────────────────────

    def create_scenario_turn(
        self,
        turn_id: str,
        session_id: str,
        turn_index: int,
        speaker_type: str,
        speaker_name: str,
        content: str,
        speaker_id: str | None = None,
        raw_response: str | None = None,
        log_request_id: str | None = None,
        anticipation: str | None = None,
    ):
        """発話ターンを作成する。

        Args:
            turn_id: ターン UUID。
            session_id: 所属プレイセッション ID。
            turn_index: セッション内の連番（0 始まり）。
            speaker_type: "user" | "narrator" | "npc" | "character"。
            speaker_name: 表示・履歴整形用のスナップショット名。
            content: 発話本文。
            speaker_id: npc 種別なら scenario_npcs.id、character 種別なら characters.id。
                        user / narrator / 未知 NPC は NULL。
            raw_response: GM の単一呼出で得たターン全体の生出力（デバッグ用）。
            log_request_id: debug_log_entries.request_id との紐付け。再生成時に引き継ぐ。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import Scenario, ScenarioSession, ScenarioTurn
            obj = ScenarioTurn(
                id=turn_id,
                session_id=session_id,
                turn_index=turn_index,
                speaker_type=speaker_type,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                content=content,
                raw_response=raw_response,
                log_request_id=log_request_id,
                anticipation=anticipation or None,
            )
            session.add(obj)
            # タイムライン封筒 dual-write（scene.turn）— 同一トランザクション。
            # 「誰のタイムラインか」はシーンに参加しているキャラクター:
            #   - うつつ（usual_days）: 世界の所有者キャラ（origin="usual"）
            #   - 通常シナリオ: pc_assignments の character 配役全員（origin="interlude"）
            # 参加キャラが1人もいないシーン（ユーザとNPCだけ）は封筒を作らない。
            sess = session.get(ScenarioSession, session_id)
            if sess is not None:
                is_usual = sess.engine_type == "usual_days"
                origin = "usual" if is_usual else "interlude"
                participant_ids: list[str] = []
                if is_usual:
                    scenario = session.get(Scenario, sess.scenario_id)
                    if scenario is not None and scenario.owner_character_id:
                        participant_ids = [scenario.owner_character_id]
                else:
                    for a in (sess.pc_assignments or []):
                        if (
                            isinstance(a, dict)
                            and a.get("player_type") == "character"
                            and a.get("character_id")
                        ):
                            participant_ids.append(str(a["character_id"]))
                for char_id in participant_ids:
                    # actor はタイムライン所有者から見た話者。自分の発話は "character"、
                    # 他キャラPCはこの世界では場の登場人物なので npc:<名前> に丸める。
                    if speaker_type == "character" and speaker_id == char_id:
                        actor = "character"
                    elif speaker_type == "user":
                        actor = "user"
                    elif speaker_type == "narrator":
                        actor = "narrator"
                    else:
                        actor = f"npc:{speaker_name}"
                    self._append_timeline_event(
                        session,
                        character_id=char_id,
                        event_type="scene.turn",
                        actor=actor,
                        origin=origin,
                        session_id=session_id,
                        source_table="scenario_turns",
                        source_id=turn_id,
                    )
            session.commit()
            session.refresh(obj)
            return obj

    def list_scenario_turns(self, session_id: str) -> list:
        """セッション内の全ターンを turn_index 昇順で返す。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioTurn
            return (
                session.query(ScenarioTurn)
                .filter(ScenarioTurn.session_id == session_id)
                .order_by(ScenarioTurn.turn_index.asc(), ScenarioTurn.created_at.asc())
                .all()
            )

    def get_unchronicled_usual_turns_for_character(self, character_id: str) -> list:
        """chronicle 用: 対象キャラのうつつ世界の未処理ターンを時系列で返す（スケジューラ用）。

        ``owner_character_id`` 一致のうつつシナリオ → ``engine_type="usual_days"`` セッション群の
        ScenarioTurn のうち ``chronicled_at IS NULL`` を ``created_at`` 昇順で返す。
        ChatMessage 側の ``get_unchronicled_messages_for_character`` のうつつ版に相当する。
        うつつ世界を持たないキャラは空リストになる。

        Args:
            character_id: キャラクターの UUID（うつつシナリオの owner_character_id）。

        Returns:
            未処理ターン一覧（時系列昇順）。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import (
                Scenario, ScenarioSession, ScenarioTurn,
            )
            return (
                session.query(ScenarioTurn)
                .join(ScenarioSession, ScenarioTurn.session_id == ScenarioSession.id)
                .join(Scenario, ScenarioSession.scenario_id == Scenario.id)
                .filter(
                    Scenario.owner_character_id == character_id,
                    ScenarioSession.engine_type == "usual_days",
                    ScenarioTurn.chronicled_at == None,  # noqa: E711
                )
                .order_by(ScenarioTurn.created_at.asc())
                .all()
            )

    def get_usual_turns_for_character_on_date(
        self, character_id: str, date_start: datetime, date_end: datetime
    ) -> list:
        """chronicle 用（target_date 経路）: 指定日のうつつターンを時系列で返す。

        ``get_messages_for_character_on_date`` のうつつ版。``chronicled_at`` の有無に
        関わらず ``created_at`` が指定日の範囲内のターンを返す（再蒸留を許す）。

        Args:
            character_id: キャラクターの UUID（うつつシナリオの owner_character_id）。
            date_start: 対象日の開始 datetime（inclusive）。
            date_end: 対象日の終了 datetime（exclusive）。

        Returns:
            当日のうつつターン一覧（時系列昇順）。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import (
                Scenario, ScenarioSession, ScenarioTurn,
            )
            return (
                session.query(ScenarioTurn)
                .join(ScenarioSession, ScenarioTurn.session_id == ScenarioSession.id)
                .join(Scenario, ScenarioSession.scenario_id == Scenario.id)
                .filter(
                    Scenario.owner_character_id == character_id,
                    ScenarioSession.engine_type == "usual_days",
                    ScenarioTurn.created_at >= date_start,
                    ScenarioTurn.created_at < date_end,
                )
                .order_by(ScenarioTurn.created_at.asc())
                .all()
            )

    def list_trpg_session_ids_by_character(self) -> dict[str, list[str]]:
        """全 ensemble_pc セッションを 1 回だけスキャンして
        ``{character_id: [session_id, ...]}`` の辞書を返す（バッチ用）。

        ``run_pending_chronicles`` のように全キャラに対して順に
        ``get_unchronicled_trpg_turns_for_character`` を呼ぶ経路では、その都度
        全 ensemble_pc セッションを Python 側で走査することになりキャラ数 ×
        セッション数の二次オーダーで重くなる（findings #13）。バッチ頭でこの
        メソッドを 1 回呼べば 1 スキャンで済み、結果を ``prefetched_session_ids``
        引数として各キャラ呼び出しに渡すことでネスト走査を回避できる。

        Returns:
            ``{character_id: [session_id, ...]}``。値リストの順序はクエリ返却順。
            該当しないキャラはキーが存在しない（呼び出し側は ``.get(char_id, [])`` で扱う）。
        """
        from backend.repositories.sqlite.store import ScenarioSession
        with self.get_session() as session:
            sessions = (
                session.query(ScenarioSession)
                .filter(ScenarioSession.engine_type == "ensemble_pc")
                .all()
            )
            result: dict[str, list[str]] = {}
            for s in sessions:
                seen_in_session: set[str] = set()
                for a in (s.pc_assignments or []):
                    if not (isinstance(a, dict) and a.get("player_type") == "character"):
                        continue
                    cid = a.get("character_id")
                    if not cid or cid in seen_in_session:
                        continue
                    seen_in_session.add(cid)
                    result.setdefault(cid, []).append(s.id)
            return result

    def _matching_ensemble_pc_session_ids(self, session, character_id: str) -> list[str]:
        """対象キャラが PC として参加している ``ensemble_pc`` セッションの ID 集合を返す。

        ``pc_assignments`` は JSON 列のため SQL では中身を直接フィルタできない。
        該当 engine_type のセッションを全件ロードして Python 側で
        ``player_type == "character"`` かつ ``character_id`` 一致のスロットを持つ
        セッションだけ拾う。``get_unchronicled_trpg_turns_for_character`` と
        ``get_trpg_turns_for_character_on_date`` の共通前処理。

        Args:
            session: 現在の SQLAlchemy セッション（呼び出し側の context 内で使う）。
            character_id: キャラクターの UUID。

        Returns:
            条件にマッチしたセッション ID のリスト。順序は session.query の返却順。
        """
        from backend.repositories.sqlite.store import ScenarioSession
        sessions = (
            session.query(ScenarioSession)
            .filter(ScenarioSession.engine_type == "ensemble_pc")
            .all()
        )
        matching: list[str] = []
        for s in sessions:
            for a in (s.pc_assignments or []):
                if (
                    isinstance(a, dict)
                    and a.get("player_type") == "character"
                    and a.get("character_id") == character_id
                ):
                    matching.append(s.id)
                    break
        return matching

    def _fetch_trpg_turns_with_title(
        self, session, session_ids: list[str], extra_filters: list
    ) -> list:
        """指定セッション ID 群の ScenarioTurn を Scenario.title と JOIN して取得し、
        ターン ORM に ``scenario_title`` 動的属性を貼り付けて返す共通処理。

        ``get_unchronicled_trpg_turns_for_character`` / ``get_trpg_turns_for_character_on_date``
        の共通取得部。呼び出し側は ``extra_filters`` でターン側の追加 WHERE 条件
        （chronicled_at IS NULL や created_at 範囲）を渡す。

        Args:
            session: 現在の SQLAlchemy セッション。
            session_ids: 取得対象の ScenarioSession ID 群。
            extra_filters: ScenarioTurn 側に AND 結合する SQLAlchemy 式のリスト。

        Returns:
            時系列昇順のターンリスト。各要素は ``scenario_title`` 動的属性付きで、
            session から expunge 済み。
        """
        from backend.repositories.sqlite.store import (
            Scenario, ScenarioSession, ScenarioTurn,
        )
        rows = (
            session.query(ScenarioTurn, Scenario.title)
            .join(ScenarioSession, ScenarioTurn.session_id == ScenarioSession.id)
            .join(Scenario, ScenarioSession.scenario_id == Scenario.id)
            .filter(
                ScenarioTurn.session_id.in_(session_ids),
                *extra_filters,
            )
            .order_by(ScenarioTurn.created_at.asc())
            .all()
        )
        result: list = []
        for turn, title in rows:
            turn.scenario_title = title or ""
            result.append(turn)
        session.expunge_all()
        return result

    def get_unchronicled_trpg_turns_for_character(
        self,
        character_id: str,
        prefetched_session_ids: list[str] | None = None,
    ) -> list:
        """chronicle 用: 対象キャラが TRPG (engine_type="ensemble_pc") の PC として参加した
        セッションの未処理ターンを、シナリオタイトル付きで時系列に返す。

        対象セッションの抽出条件:
          - ``engine_type == "ensemble_pc"``
          - ``pc_assignments`` に ``player_type == "character"`` かつ ``character_id`` 一致の
            スロットが含まれる

        セッション抽出と turn 取得は共通ヘルパ
        ``_matching_ensemble_pc_session_ids`` / ``_fetch_trpg_turns_with_title``
        に集約している。本メソッドは「未処理 (chronicled_at IS NULL)」の filter だけ提供する。

        Args:
            character_id: キャラクターの UUID。
            prefetched_session_ids: バッチ呼び出し時の最適化用。あらかじめ
                ``list_trpg_session_ids_by_character`` で取得した該当キャラの
                session_id リストを渡すと、内部の全 ensemble_pc セッションスキャンを
                スキップする（findings #13）。

        Returns:
            未処理 TRPG ターンのリスト（時系列昇順）。各要素には ``scenario_title``
            属性が付いている。
        """
        from backend.repositories.sqlite.store import ScenarioTurn
        with self.get_session() as session:
            if prefetched_session_ids is None:
                matching_session_ids = self._matching_ensemble_pc_session_ids(session, character_id)
            else:
                matching_session_ids = prefetched_session_ids
            if not matching_session_ids:
                return []
            return self._fetch_trpg_turns_with_title(
                session,
                matching_session_ids,
                [ScenarioTurn.chronicled_at == None],  # noqa: E711
            )

    def get_trpg_turns_for_character_on_date(
        self,
        character_id: str,
        date_start: datetime,
        date_end: datetime,
        prefetched_session_ids: list[str] | None = None,
    ) -> list:
        """chronicle 用 (target_date 経路): 対象キャラが PC 参加した TRPG セッションの
        指定日ターンをシナリオタイトル付きで時系列に返す。

        ``get_unchronicled_trpg_turns_for_character`` の target_date 版。
        ``chronicled_at`` の有無に関わらず ``created_at`` が指定日範囲のターンを返す
        （再蒸留を許す）。

        Args:
            character_id: キャラクターの UUID。
            date_start: 対象日の開始 datetime (inclusive)。
            date_end: 対象日の終了 datetime (exclusive)。
            prefetched_session_ids: バッチ呼び出し時の最適化用。事前計算済みの
                該当 session_id 群を渡すと、内部の全 ensemble_pc セッションスキャンを
                スキップする（findings #13）。

        Returns:
            当日の TRPG ターン一覧（時系列昇順）。各要素には ``scenario_title``
            属性が付いている。
        """
        from backend.repositories.sqlite.store import ScenarioTurn
        with self.get_session() as session:
            if prefetched_session_ids is None:
                matching_session_ids = self._matching_ensemble_pc_session_ids(session, character_id)
            else:
                matching_session_ids = prefetched_session_ids
            if not matching_session_ids:
                return []
            return self._fetch_trpg_turns_with_title(
                session,
                matching_session_ids,
                [
                    ScenarioTurn.created_at >= date_start,
                    ScenarioTurn.created_at < date_end,
                ],
            )

    def mark_scenario_turns_as_chronicled(self, turn_ids: list[str]) -> None:
        """指定 ScenarioTurn の chronicled_at を現在日時にセットする。

        ChatMessage 側の ``mark_messages_as_chronicled`` のうつつ版。Chronicle で
        当日会話へ合流させたうつつターンを「処理済み」にして二重処理を防ぐ。

        Args:
            turn_ids: 処理済みにする ScenarioTurn ID のリスト。
        """
        if not turn_ids:
            return
        from datetime import datetime as dt
        now = dt.utcnow()
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioTurn
            session.query(ScenarioTurn).filter(
                ScenarioTurn.id.in_(turn_ids)
            ).update({"chronicled_at": now}, synchronize_session=False)
            session.commit()

    def delete_scenario_turns_from(self, session_id: str, turn_id: str) -> bool:
        """指定 turn_id 以降（自身を含む）のターンをまとめて削除する。

        ユーザ発話の編集・GM ターンの再生成で使う「区切り点以降を一掃する」操作。
        編集・再生成パターンは既存 chat の `delete_chat_messages_from` と同じ思想。

        副作用: 削除域に `synopsis_last_turn_index` が含まれる場合、その値を
        `pivot.turn_index - 1` までクランプする。クランプを怠ると、ロールバック後の
        セッションで「すでに削除されたターンまで蒸留済み」という誤認識が残り、
        以降の `maybe_update_auto_synopsis` が `new_dropped` 空判定で永久に
        skip される（あらすじが二度と再生成されない）バグになる。

        Args:
            session_id: 対象セッション ID。
            turn_id: この turn と、これより後（turn_index が大きいもの）を全削除する。

        Returns:
            削除を実行した場合は True、turn_id が見つからなければ False。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioSession, ScenarioTurn
            pivot = session.get(ScenarioTurn, turn_id)
            if pivot is None or pivot.session_id != session_id:
                return False
            pivot_index = int(pivot.turn_index)
            # 封筒は削除せず retracted マーク（不可逆性の担保。データは残す）
            ids_to_delete = [
                row[0]
                for row in session.query(ScenarioTurn.id)
                .filter(
                    ScenarioTurn.session_id == session_id,
                    ScenarioTurn.turn_index >= pivot_index,
                )
                .all()
            ]
            self._retract_timeline_events_in_session(
                session, "scenario_turns", ids_to_delete
            )
            session.query(ScenarioTurn).filter(
                ScenarioTurn.session_id == session_id,
                ScenarioTurn.turn_index >= pivot_index,
            ).delete(synchronize_session=False)
            # synopsis_last_turn_index のクランプ（ロールバック整合性）
            sess_obj = session.get(ScenarioSession, session_id)
            if (
                sess_obj is not None
                and sess_obj.synopsis_last_turn_index is not None
                and int(sess_obj.synopsis_last_turn_index) >= pivot_index
            ):
                sess_obj.synopsis_last_turn_index = pivot_index - 1
                sess_obj.updated_at = datetime.now()
            session.commit()
            return True

    def get_next_scenario_turn_index(self, session_id: str) -> int:
        """次に使うべき turn_index を返す（既存最大値+1、なければ 0）。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioTurn
            row = (
                session.query(ScenarioTurn.turn_index)
                .filter(ScenarioTurn.session_id == session_id)
                .order_by(ScenarioTurn.turn_index.desc())
                .first()
            )
            if row is None:
                return 0
            return int(row[0]) + 1
