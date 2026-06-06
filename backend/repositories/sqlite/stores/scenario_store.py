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
        user_alias: str,
        scenario: str | None = None,
        intro: str | None = None,
        history_max_turns: int | None = None,
        history_max_chars: int | None = None,
        custom_system_prompt: str | None = None,
        dice_pool_spec: dict | None = None,
        pc_slots: list[dict] | None = None,
    ):
        """シナリオテンプレートを新規作成する。

        場所・空気・語り口・テンポなどはすべて `scenario` テキストにまとめて記述する。
        intro はセッション開始時に固定ターンとして挿入される導入部（@キャラ: 記法）。
        custom_system_prompt はGMシステムプロンプトの完全カスタマイズ。
                         空の場合、デフォルトテンプレートが自動設定される。
        dice_pool_spec は ensemble_pc エンジン時に毎ターン乱数生成する種別と本数の dict。
                         例: {"d6": 10, "d100": 5}。NULL なら engine 側既定値 {"d6": 10}。
        pc_slots は ensemble_pc エンジン時の PC枠定義。
                         [{"slot_id":"pc1","name":"アリス","description":"剣士。商家の出。..."}]。
                         シナリオ側で人物像・知っていることを含めて記述する。
                         セッション開始時に各枠を「ユーザが演じる/AIキャラが演じる」と割り振る。

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
                user_alias=user_alias,
                history_max_turns=history_max_turns,
                history_max_chars=history_max_chars,
                custom_system_prompt=custom_system_prompt,
                dice_pool_spec=dice_pool_spec,
                pc_slots=pc_slots,
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

    def list_scenarios(self, limit: int = 100) -> list:
        """シナリオテンプレート一覧を更新日時の新しい順で返す。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import Scenario
            return (
                session.query(Scenario)
                .order_by(Scenario.updated_at.desc())
                .limit(limit)
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

    def list_scenario_sessions(self, limit: int = 100) -> list:
        """プレイセッション一覧を更新日時の新しい順で返す。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioSession
            return (
                session.query(ScenarioSession)
                .order_by(ScenarioSession.updated_at.desc())
                .limit(limit)
                .all()
            )

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
            from backend.repositories.sqlite.store import ScenarioTurn
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
            session.commit()
            session.refresh(obj)
            return obj

    def update_scenario_turn_log_request_id(
        self, turn_id: str, log_request_id: str
    ) -> None:
        """発話ターンの log_request_id を更新する。

        Args:
            turn_id: 更新対象ターンの UUID。
            log_request_id: セットする debug_log_entries.request_id。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ScenarioTurn
            obj = session.get(ScenarioTurn, turn_id)
            if obj:
                obj.log_request_id = log_request_id
                session.commit()

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
