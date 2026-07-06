"""SQLiteStore の冪等マイグレーション群。

既存 DB を現行 ORM スキーマ（models.py）へ追従させる処理を
SQLiteMigrationsMixin に集約する。
全メソッドは冪等であり、起動時（SQLiteStore.__init__）に毎回呼ばれても安全。
新規 DB は ``Base.metadata.create_all`` が定義通りに作成するため、
ここのマイグレーションは「旧スキーマの既存 DB」にのみ実質的な変更を行う。
"""


class SQLiteMigrationsMixin:
    """既存 DB スキーマを現行定義へ移行する冪等マイグレーションの Mixin。

    `self.engine` を持つクラス（SQLiteStore）に多重継承で合成される前提。
    各メソッドは SQLiteStore.__init__ から テーブル作成後に呼び出される。
    """

    def _migrate_drop_session_drifts(self) -> None:
        """SELF_DRIFT 機能撤去に伴い session_drifts テーブルを削除する。

        ドリフト機能（drift_manager.py / chat_drifts.py / drift ツール）は撤去済みのため、
        既存 DB にのみ残る session_drifts テーブルを物理削除する。
        新規 DB には ORM 定義が無く作成されないため、本マイグレーションは冪等。
        """
        with self.engine.begin() as conn:
            conn.exec_driver_sql("DROP TABLE IF EXISTS session_drifts")

    def _migrate_unify_user_alias_to_pc_slot(self) -> None:
        """`scenarios.user_alias` を廃止し、ユーザPCを PC Slot へ一本化する。

        旧スキーマ: ユーザの呼称は scenarios.user_alias（単一・NOT NULL）。
        新スキーマ: ユーザも 1 つの PC枠（scenarios.pc_slots の 1 要素）として表現し、
                    セッションの pc_assignments で player_type="user" を割り当てる。
                    user_alias 列は廃止する。

        移行手順（冪等）:
            1. user_alias 列が無ければ移行済み → 何もしない。
            2. ensemble テンプレ（pc_slots 空）は user_alias 名の user 枠を新規作成して
               pc_slots へ入れる（呼称を失わないため）。
            3. player_type="user" の割当が無いセッションには、user 枠への割当
               {"slot_id":..., "player_type":"user"} を追加する。user 枠が無ければ
               user_alias 名で作成（既存に同名枠があれば再利用）。
               ※ ensemble_pc テンプレで既に全セッションがユーザ枠を持つ場合は触らない
                 （余計な枠を生やさない）。
            4. scenarios.user_alias 列を DROP COLUMN する（SQLite 3.35+）。

        新規 DB（ORM に user_alias 列が無い）では 1 で抜けるため冪等。
        """
        import json

        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "scenarios" not in tables:
                return
            scen_cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(scenarios)"
                ).fetchall()
            }
            if "user_alias" not in scen_cols:
                return  # 既に移行済み
            has_sessions_table = "scenario_sessions" in tables

            def _unify_legacy_gm_prompt(text: str) -> str:
                """既存 custom_system_prompt から旧「主役（プレイヤー）」フレーミングを除く。

                ユーザPCを特別扱いせず全PCを均一ロスターへ集約する新方針に合わせ、
                既知の話者ブロックの主役行と、PC領分節の中の人言及を置換する。
                ユーザが編集して文言がずれている場合は一致せず no-op（ベストエフォート）。
                """
                replacements = [
                    # 既知の話者ブロックの「主役」行を丸ごと削除（後続改行ごと）
                    (
                        "@{user_alias}   ← この物語の主役（プレイヤー）。"
                        "あなたは絶対に代弁しない。\n",
                        "",
                    ),
                    # PC配役ブロック見出しを新名称へ
                    ("# PC配役（プレイヤーキャラクター）", "# プレイヤーキャラクター（PC）"),
                    # NPC 紹介の二重掲載を解消（既知の話者側の簡潔リストを廃し NPC詳細へ一本化）
                    (
                        "{npcs_summary}",
                        "（NPC の顔ぶれと人物像は下記「NPC詳細」を参照）",
                    ),
                    # PC領分節の user_alias 参照と中の人言及を中立化
                    (
                        "ここで言う **PC** は、@{user_alias} と「PC配役」"
                        "セクションに掲げた全員を指します。\n"
                        "PC はそれぞれ別の人格（ユーザ本人または別の AI キャラクター）"
                        "が演じます。",
                        "ここで言う **PC** は、「プレイヤーキャラクター（PC）」"
                        "セクションに掲げた全員を指します。\n"
                        "PC はそれぞれ別の人格が演じます"
                        "（あなた＝GM はその中身が人間か AI かを意識しません）。",
                    ),
                ]
                for old, new in replacements:
                    text = text.replace(old, new)
                return text

            scen_rows = conn.exec_driver_sql(
                "SELECT id, user_alias, pc_slots, custom_system_prompt FROM scenarios"
            ).fetchall()
            for sid, alias, pc_slots_raw, csp_raw in scen_rows:
                alias = (str(alias or "").strip()) or "プレイヤー"
                try:
                    pc_slots = json.loads(pc_slots_raw) if pc_slots_raw else []
                    if not isinstance(pc_slots, list):
                        pc_slots = []
                except (json.JSONDecodeError, TypeError):
                    pc_slots = []

                # pc_slots の永続化は変更があったときだけ行うためのフラグ。
                slots_dirty = False

                def ensure_user_slot() -> str:
                    """user_alias 名の PC枠を確保して slot_id を返す（必要時のみ生成）。"""
                    nonlocal slots_dirty
                    existing = next(
                        (
                            s for s in pc_slots
                            if isinstance(s, dict)
                            and str(s.get("name", "")).strip() == alias
                        ),
                        None,
                    )
                    if existing is not None:
                        return str(existing.get("slot_id"))
                    existing_ids = {
                        str(s.get("slot_id"))
                        for s in pc_slots if isinstance(s, dict)
                    }
                    new_id = "user"
                    n = 2
                    while new_id in existing_ids:
                        new_id = f"user{n}"
                        n += 1
                    pc_slots.insert(0, {
                        "slot_id": new_id, "name": alias, "description": "",
                    })
                    slots_dirty = True
                    return new_id

                user_slot_id: str | None = None
                # 2. ensemble テンプレ（pc_slots 空）は呼称保全のため user 枠を作る
                if not pc_slots:
                    user_slot_id = ensure_user_slot()

                # 3. ユーザ割当の無いセッションへ user 割当を補う
                if has_sessions_table:
                    sess_rows = conn.exec_driver_sql(
                        "SELECT id, pc_assignments FROM scenario_sessions "
                        "WHERE scenario_id = ?",
                        (sid,),
                    ).fetchall()
                    for sess_id, asn_raw in sess_rows:
                        try:
                            asn = json.loads(asn_raw) if asn_raw else []
                            if not isinstance(asn, list):
                                asn = []
                        except (json.JSONDecodeError, TypeError):
                            asn = []
                        has_user = any(
                            isinstance(a, dict)
                            and str(a.get("player_type", "")) == "user"
                            for a in asn
                        )
                        if has_user:
                            continue
                        if user_slot_id is None:
                            user_slot_id = ensure_user_slot()
                        asn.append({
                            "slot_id": user_slot_id, "player_type": "user",
                        })
                        conn.exec_driver_sql(
                            "UPDATE scenario_sessions SET pc_assignments = ? "
                            "WHERE id = ?",
                            (json.dumps(asn, ensure_ascii=False), sess_id),
                        )

                if slots_dirty:
                    conn.exec_driver_sql(
                        "UPDATE scenarios SET pc_slots = ? WHERE id = ?",
                        (json.dumps(pc_slots, ensure_ascii=False), sid),
                    )

                # 既存 custom_system_prompt の旧「主役」フレーミングを除去する（ベストエフォート）。
                if csp_raw:
                    new_csp = _unify_legacy_gm_prompt(csp_raw)
                    if new_csp != csp_raw:
                        conn.exec_driver_sql(
                            "UPDATE scenarios SET custom_system_prompt = ? WHERE id = ?",
                            (new_csp, sid),
                        )

            # 4. 旧 user_alias 列を削除（SQLite 3.35+）
            conn.exec_driver_sql("ALTER TABLE scenarios DROP COLUMN user_alias")

    def _migrate_gm_preset_id_to_session(self) -> None:
        """`gm_preset_id` を scenarios → scenario_sessions に移行する。

        旧スキーマ: scenarios.gm_preset_id（NOT NULL）にシナリオ単位で保持
        新スキーマ: scenario_sessions.gm_preset_id（NOT NULL）にセッション単位で保持

        移行手順:
            1. scenario_sessions に gm_preset_id 列が無ければ ADD COLUMN
            2. scenarios.gm_preset_id を JOIN で各セッションへバックフィル
            3. scenarios.gm_preset_id を DROP COLUMN（SQLite 3.35+ が必要）

        新規 DB（既に新スキーマ）では何もしない。冪等。
        """
        def _columns_of(conn, table: str) -> set[str]:
            rows = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
            return {r[1] for r in rows}

        with self.engine.begin() as conn:
            # 両テーブルが存在するか（最初の起動時は scenarios すら無いこともある）
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "scenarios" not in tables or "scenario_sessions" not in tables:
                return

            session_cols = _columns_of(conn, "scenario_sessions")
            scenario_cols = _columns_of(conn, "scenarios")

            # 1. scenario_sessions に列を追加（既に新スキーマなら skip）
            if "gm_preset_id" not in session_cols:
                conn.exec_driver_sql(
                    "ALTER TABLE scenario_sessions "
                    "ADD COLUMN gm_preset_id TEXT NOT NULL DEFAULT ''"
                )

            # 2. 旧 scenarios.gm_preset_id が残っていればバックフィル
            if "gm_preset_id" in scenario_cols:
                conn.exec_driver_sql(
                    "UPDATE scenario_sessions "
                    "SET gm_preset_id = COALESCE(("
                    "  SELECT s.gm_preset_id FROM scenarios s "
                    "  WHERE s.id = scenario_sessions.scenario_id"
                    "), gm_preset_id) "
                    "WHERE (gm_preset_id IS NULL OR gm_preset_id = '')"
                )
                # 3. 旧列を削除（SQLite 3.35+）
                conn.exec_driver_sql(
                    "ALTER TABLE scenarios DROP COLUMN gm_preset_id"
                )

    def _migrate_add_synopsis_preset_id(self) -> None:
        """`scenario_sessions` に `synopsis_preset_id` 列を追加する。

        旧スキーマ: あらすじ蒸留はセッションの `gm_preset_id` を使い回す
        新スキーマ: あらすじ蒸留専用の `synopsis_preset_id` を持つ

        移行手順:
            1. scenario_sessions に列が無ければ ADD COLUMN
            2. 既存行は `gm_preset_id` を初期値として埋める（従来挙動を維持）

        新規 DB（既に新スキーマ）では何もしない。冪等。
        """
        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "scenario_sessions" not in tables:
                return
            cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(scenario_sessions)"
                ).fetchall()
            }
            if "synopsis_preset_id" in cols:
                return
            conn.exec_driver_sql(
                "ALTER TABLE scenario_sessions "
                "ADD COLUMN synopsis_preset_id TEXT NOT NULL DEFAULT ''"
            )
            # 既存行は gm_preset_id をそのままコピー（従来挙動を維持）
            conn.exec_driver_sql(
                "UPDATE scenario_sessions "
                "SET synopsis_preset_id = gm_preset_id "
                "WHERE (synopsis_preset_id IS NULL OR synopsis_preset_id = '')"
            )

    def _migrate_add_preset_timeout_seconds(self) -> None:
        """`llm_model_presets` テーブルに `timeout_seconds` 列を追加する。

        プロバイダーAPIリクエストのタイムアウトをプリセット単位で指定するための列。
        既存DBでは列が存在しないため ALTER TABLE で追加し、デフォルト300秒（5分）を入れる。
        新規DB（既に新スキーマ）では何もしない。冪等。
        """
        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "llm_model_presets" not in tables:
                return
            cols = {
                r[1]
                for r in conn.exec_driver_sql("PRAGMA table_info(llm_model_presets)").fetchall()
            }
            if "timeout_seconds" in cols:
                return
            conn.exec_driver_sql(
                "ALTER TABLE llm_model_presets "
                "ADD COLUMN timeout_seconds INTEGER NOT NULL DEFAULT 300"
            )

    def _migrate_add_debug_log_entries(self) -> None:
        """`debug_log_entries` テーブルが存在しない既存 DB への互換マイグレーション。

        `Base.metadata.create_all` が新規テーブルを作るが、既存 DB には
        インデックスが追加されない場合があるため、インデックスだけ別途作成する。
        新規 DB（既に新スキーマ）では何もしない。冪等。
        """
        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "debug_log_entries" not in tables:
                # テーブルがない場合は create_all で作成済みのはずだが念のため
                return
            # request_id インデックスが存在しなければ作成（冪等）
            indexes = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND tbl_name='debug_log_entries'"
                ).fetchall()
            }
            if "ix_debug_log_entries_request_id" not in indexes:
                conn.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS ix_debug_log_entries_request_id "
                    "ON debug_log_entries (request_id)"
                )

    def _migrate_add_scenario_turn_log_request_id(self) -> None:
        """`scenario_turns` に `log_request_id` 列を追加する。

        再生成時に同一 request_id を引き継ぐためのカラム。
        既存 DB には列がないため ALTER TABLE で追加する。冪等。
        """
        with self.engine.begin() as conn:
            cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(scenario_turns)"
                ).fetchall()
            }
            if "log_request_id" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE scenario_turns ADD COLUMN log_request_id TEXT"
                )

    def _migrate_add_chat_message_anticipation(self) -> None:
        """`chat_messages` に `anticipation` 列を追加する。

        キャラクターが本文末尾に書いた予想（期待）タグの抽出結果を保存する列。
        既存 DB には列がないため ALTER TABLE で追加する。冪等。
        """
        with self.engine.begin() as conn:
            cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(chat_messages)"
                ).fetchall()
            }
            if "anticipation" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE chat_messages ADD COLUMN anticipation TEXT"
                )

    def _migrate_add_scenario_turn_anticipation(self) -> None:
        """`scenario_turns` に `anticipation` 列を追加する。

        GM がターン末尾に書いた予想（期待）タグの抽出結果を保存する列。
        既存 DB には列がないため ALTER TABLE で追加する。冪等。
        """
        with self.engine.begin() as conn:
            cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(scenario_turns)"
                ).fetchall()
            }
            if "anticipation" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE scenario_turns ADD COLUMN anticipation TEXT"
                )

    def _migrate_add_scenario_turn_chronicled_at(self) -> None:
        """`scenario_turns` に `chronicled_at` 列を追加する。

        うつつ（usual_days）のやり取りを Chronicle 対象へ合流させるための列。
        ChatMessage.chronicled_at と同じく「NULL=未処理 / タイムスタンプ=処理済み」を表す。
        既存 DB には列がないため ALTER TABLE で追加する。冪等。
        """
        with self.engine.begin() as conn:
            cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(scenario_turns)"
                ).fetchall()
            }
            if "chronicled_at" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE scenario_turns ADD COLUMN chronicled_at DATETIME"
                )

    def _migrate_add_scenario_banner_data(self) -> None:
        """`scenarios` に `banner_data` 列を追加する。

        シナリオのバナー画像（base64 data URI）を保存する列。
        既存 DB には列がないため ALTER TABLE で追加する。冪等。
        """
        with self.engine.begin() as conn:
            cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(scenarios)"
                ).fetchall()
            }
            if "banner_data" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE scenarios ADD COLUMN banner_data TEXT"
                )

    def _migrate_add_memory_origin(self) -> None:
        """`inscribed_memories` と `working_memory_threads` に `origin` 列を追加する。

        Scenario PC モード（TRPG的にキャラがPCを演じるモード）で生じた記憶を
        `origin='interlude'` で識別するための列。日常体験は `origin='real'`（既定）。
        検索フィルタとキャラ本人の文脈把握に使う。
        既存DBに列がなければ ALTER TABLE で追加する。冪等。
        """
        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "inscribed_memories" in tables:
                cols = {
                    r[1]
                    for r in conn.exec_driver_sql(
                        "PRAGMA table_info(inscribed_memories)"
                    ).fetchall()
                }
                if "origin" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE inscribed_memories "
                        "ADD COLUMN origin TEXT NOT NULL DEFAULT 'real'"
                    )
            if "working_memory_threads" in tables:
                cols = {
                    r[1]
                    for r in conn.exec_driver_sql(
                        "PRAGMA table_info(working_memory_threads)"
                    ).fetchall()
                }
                if "origin" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE working_memory_threads "
                        "ADD COLUMN origin TEXT NOT NULL DEFAULT 'real'"
                    )

    def _migrate_add_scenario_pc_mode(self) -> None:
        """Scenario PC モード関連カラムを追加する。

        - `scenarios.dice_pool_spec` (JSON, NULL可): ダイスプール仕様。
        - `scenarios.pc_slots` (JSON, NULL可): PC枠定義。
        - `scenario_sessions.pc_assignments` (JSON, NULL可): PC配役一覧。

        いずれも `engine_type='ensemble_pc'` のセッション専用フィールド。
        既存DBに列がなければ ALTER TABLE で追加する。冪等。
        """
        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "scenarios" in tables:
                cols = {
                    r[1]
                    for r in conn.exec_driver_sql(
                        "PRAGMA table_info(scenarios)"
                    ).fetchall()
                }
                if "dice_pool_spec" not in cols:
                    # SQLite の JSON 型は TEXT として保存される。
                    conn.exec_driver_sql(
                        "ALTER TABLE scenarios ADD COLUMN dice_pool_spec TEXT"
                    )
                if "pc_slots" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE scenarios ADD COLUMN pc_slots TEXT"
                    )
            if "scenario_sessions" in tables:
                cols = {
                    r[1]
                    for r in conn.exec_driver_sql(
                        "PRAGMA table_info(scenario_sessions)"
                    ).fetchall()
                }
                if "pc_assignments" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE scenario_sessions ADD COLUMN pc_assignments TEXT"
                    )

    def _migrate_drop_afterglow_columns(self) -> None:
        """Afterglow（感情継続機構）廃止に伴い関連カラムを削除する。

        - `chat_sessions.afterglow_session_id`
        - `characters.afterglow_default`

        SQLite 3.35+ の DROP COLUMN を使う。新規DBには列が無いため何もしない。冪等。
        """
        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "chat_sessions" in tables:
                cols = {
                    r[1]
                    for r in conn.exec_driver_sql(
                        "PRAGMA table_info(chat_sessions)"
                    ).fetchall()
                }
                if "afterglow_session_id" in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE chat_sessions DROP COLUMN afterglow_session_id"
                    )
            if "characters" in tables:
                cols = {
                    r[1]
                    for r in conn.exec_driver_sql(
                        "PRAGMA table_info(characters)"
                    ).fetchall()
                }
                if "afterglow_default" in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE characters DROP COLUMN afterglow_default"
                    )

    def _migrate_add_scenario_custom_system_prompt(self) -> None:
        """`scenarios` に `custom_system_prompt` 列を追加し、既存シナリオに規定プロンプトを設定する。

        GMシステムプロンプトをシナリオ単位でカスタマイズするためのカラム。
        既存シナリオには DEFAULT_GM_SYSTEM_PROMPT_TEMPLATE を設定する。冪等。
        """
        with self.engine.begin() as conn:
            cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(scenarios)"
                ).fetchall()
            }
            if "custom_system_prompt" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE scenarios ADD COLUMN custom_system_prompt TEXT"
                )
                # カラム追加後、既存シナリオにデフォルトプロンプトを設定
                from backend.services.scenario_chat.prompt_builder import (
                    DEFAULT_GM_SYSTEM_PROMPT_TEMPLATE,
                )

                conn.exec_driver_sql(
                    "UPDATE scenarios SET custom_system_prompt = ? WHERE custom_system_prompt IS NULL",
                    (DEFAULT_GM_SYSTEM_PROMPT_TEMPLATE,),
                )

    def _migrate_extract_user_fields_to_character(self) -> None:
        """`characters` に `user_label` `user_position` 列を追加し、既存うつつから引き上げる。

        旧設計: ユーザの呼称・位置づけはうつつシナリオの `pc_slots[slot_id="user"]` だけにあり、
            うつつ以外（1on1・全バッチ処理）にはキャラへ届かなかった。
        新設計: `characters` テーブルを source of truth とし、1on1・全バッチ処理のシステム
            プロンプトに「あなたが対話する相手」ブロックとして注入する。うつつの pc_slots[user]
            は、`_persist_usual_world` で characters の値から毎回再構築される派生情報になる。

        移行手順（冪等）:
            1. characters に user_label / user_position 列が無ければ ADD COLUMN（NOT NULL DEFAULT ''）
            2. owner_character_id 付きうつつシナリオを走査し、pc_slots[slot_id="user"] の
               name → characters.user_label、description（不在マーカーを除去）→ characters.user_position
               を埋める。既に characters 側に値があるキャラは触らない（上書き回避）。
            3. pc_slots[user] はそのまま残す（うつつ GM 側のコードが従来通り参照できるため）。

        新規 DB（既に新スキーマ）では 1 で列が既存と判定されスキップ。冪等。
        """
        import json

        # _persist_usual_world と同じ不在マーカー。直接 import すると循環や API 層依存に
        # なるため、ここでは文字列リテラルとして同期する（変更時は両方更新）。
        absent_prefix = "【この場面に不在・姿/言動を描かない】"

        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "characters" not in tables:
                return
            char_cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(characters)"
                ).fetchall()
            }
            added = False
            if "user_label" not in char_cols:
                conn.exec_driver_sql(
                    "ALTER TABLE characters "
                    "ADD COLUMN user_label TEXT NOT NULL DEFAULT ''"
                )
                added = True
            if "user_position" not in char_cols:
                conn.exec_driver_sql(
                    "ALTER TABLE characters "
                    "ADD COLUMN user_position TEXT NOT NULL DEFAULT ''"
                )
                added = True
            if not added:
                return  # 既に移行済み

            # うつつシナリオから引き上げ（characters 側が空のキャラのみ）。
            if "scenarios" not in tables:
                return
            scen_cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(scenarios)"
                ).fetchall()
            }
            if "owner_character_id" not in scen_cols or "pc_slots" not in scen_cols:
                return
            rows = conn.exec_driver_sql(
                "SELECT owner_character_id, pc_slots FROM scenarios "
                "WHERE owner_character_id IS NOT NULL"
            ).fetchall()
            for owner_id, pc_slots_raw in rows:
                if not owner_id:
                    continue
                try:
                    pc_slots = json.loads(pc_slots_raw) if pc_slots_raw else []
                    if not isinstance(pc_slots, list):
                        continue
                except (json.JSONDecodeError, TypeError):
                    continue
                user_slot = next(
                    (
                        s for s in pc_slots
                        if isinstance(s, dict) and s.get("slot_id") == "user"
                    ),
                    None,
                )
                if user_slot is None:
                    continue
                label = (str(user_slot.get("name") or "")).strip()
                desc = str(user_slot.get("description") or "")
                if desc.startswith(absent_prefix):
                    desc = desc[len(absent_prefix):].strip()
                position = desc.strip()
                if not (label or position):
                    continue
                # キャラ側の既存値が空のときだけ上書き（手動で別の値を入れた可能性を尊重）。
                existing = conn.exec_driver_sql(
                    "SELECT user_label, user_position FROM characters WHERE id = ?",
                    (owner_id,),
                ).fetchone()
                if existing is None:
                    continue
                ex_label, ex_pos = existing
                if (ex_label or "").strip() or (ex_pos or "").strip():
                    continue
                conn.exec_driver_sql(
                    "UPDATE characters SET user_label = ?, user_position = ? "
                    "WHERE id = ?",
                    (label, position, owner_id),
                )

    def _migrate_add_user_visibility_note(self) -> None:
        """`characters` に `user_visibility_note` 列を追加する。

        うつつ世界の GM へ「キャラ本人がユーザを周囲にどう伝えているか」を流し込む素材。
        空（既定）なら NPC はユーザを話題にしない（完全秘匿）。非空ならその文面が
        GM プロンプトの「不在の関係者」ブロックに素通しで載り、NPC の自発的言及の手がかりになる。
        既存 DB には列が無いため ALTER TABLE で追加する。冪等。
        """
        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "characters" not in tables:
                return
            cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(characters)"
                ).fetchall()
            }
            if "user_visibility_note" in cols:
                return
            conn.exec_driver_sql(
                "ALTER TABLE characters "
                "ADD COLUMN user_visibility_note TEXT NOT NULL DEFAULT ''"
            )

    def _migrate_add_usual_days(self) -> None:
        """うつつ（Usual Days）用カラムを `scenarios` に追加する。

        - `scenarios.owner_character_id` (String, NULL可): うつつ世界の所有者キャラ ID。
          NULL=汎用シナリオ、値あり=そのキャラのうつつ世界（汎用一覧から除外する判定キー）。
        - `scenarios.usual_config` (JSON, NULL可): うつつ運用設定（有効化トグル・スロット時刻・
          時間グリッド・偶発イベントカテゴリ・発生率・1シーン上限ターン・GM/PCプリセット）。

        SQLite の JSON 型は TEXT として保存される。
        既存DBに列がなければ ALTER TABLE で追加する。新規DBは ORM 定義で作成済みのため
        列が既にあり何もしない。冪等。
        """
        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "scenarios" not in tables:
                return
            cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(scenarios)"
                ).fetchall()
            }
            if "owner_character_id" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE scenarios ADD COLUMN owner_character_id TEXT"
                )
            if "usual_config" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE scenarios ADD COLUMN usual_config TEXT"
                )

    def _migrate_drop_group_chat(self) -> None:
        """グループチャット機能撤去に伴い、関連レコードとカラムを物理削除する。

        撤去対象:
            - chat_messages のうち session_type='group' に属する行
            - chat_images のうち session_type='group' に属する行
            - chat_sessions のうち session_type='group' の行
            - chat_sessions.group_config 列（SQLite 3.35+ の DROP COLUMN）

        Director 廃止・GroupChat 統合に伴う Step1。group_config 列の有無で冪等性を担保する。
        既に列が無ければ移行済みとして何もしない。
        """
        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "chat_sessions" not in tables:
                return
            cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(chat_sessions)"
                ).fetchall()
            }
            if "group_config" not in cols:
                return  # 既に移行済み

            # 関連レコード削除（外部キーに頼らず、明示的に順序を制御する）
            if "chat_messages" in tables:
                conn.exec_driver_sql(
                    "DELETE FROM chat_messages "
                    "WHERE session_id IN ("
                    "  SELECT id FROM chat_sessions WHERE session_type='group'"
                    ")"
                )
            if "chat_images" in tables:
                conn.exec_driver_sql(
                    "DELETE FROM chat_images "
                    "WHERE session_id IN ("
                    "  SELECT id FROM chat_sessions WHERE session_type='group'"
                    ")"
                )
            conn.exec_driver_sql(
                "DELETE FROM chat_sessions WHERE session_type='group'"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_sessions DROP COLUMN group_config"
            )

    def _migrate_add_pressure_profile(self) -> None:
        """`characters` に `pressure_profile` 列を追加する（めぐり Phase 3）。

        圧力（社会圧・退屈圧・体調圧）の体質プロファイル JSON。
        NULL = 標準プロファイル。既存 DB には列が無いため ALTER TABLE で追加する。冪等。
        """
        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "characters" not in tables:
                return
            cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(characters)"
                ).fetchall()
            }
            if "pressure_profile" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE characters ADD COLUMN pressure_profile TEXT"
                )

    def _migrate_add_gate_columns(self) -> None:
        """応答可能性ゲート（めぐり Phase 5）関連カラムを追加する。

        - characters.availability_schedule (JSON, NULL可): 生活時間割（応答不可時間帯）。
        - characters.away_until (DATETIME, NULL可): 動的不在の期限。
        - characters.away_reason (TEXT, NULL可): 不在理由。
        - chat_messages.delivered_at (DATETIME, NULL可): メッセージ預かりマーカー。
          既存行は「配達済み」として created_at をバックフィルする
          （NULL のまま残すと過去メッセージが預かり中扱いになるため）。

        既存 DB に列がなければ ALTER TABLE で追加する。冪等。
        """
        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "characters" in tables:
                cols = {
                    r[1]
                    for r in conn.exec_driver_sql(
                        "PRAGMA table_info(characters)"
                    ).fetchall()
                }
                if "availability_schedule" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE characters ADD COLUMN availability_schedule TEXT"
                    )
                if "action_menu" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE characters ADD COLUMN action_menu TEXT"
                    )
                if "away_until" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE characters ADD COLUMN away_until DATETIME"
                    )
                if "away_reason" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE characters ADD COLUMN away_reason TEXT"
                    )
            if "chat_messages" in tables:
                cols = {
                    r[1]
                    for r in conn.exec_driver_sql(
                        "PRAGMA table_info(chat_messages)"
                    ).fetchall()
                }
                if "delivered_at" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE chat_messages ADD COLUMN delivered_at DATETIME"
                    )
                    # 既存メッセージは配達済み扱い（created_at で埋める）
                    conn.exec_driver_sql(
                        "UPDATE chat_messages SET delivered_at = created_at "
                        "WHERE delivered_at IS NULL"
                    )

    def _migrate_backfill_timeline_events(self) -> None:
        """タイムライン封筒（timeline_events）へ過去データをバックフィルする。

        めぐり（巡り / Aliveness）導入時の一回きりの移行（docs/aliveness_plan.md §2.1）。
        テーブル自体は ``Base.metadata.create_all`` が作成済みの前提で、
        既存テーブルの過去行から封筒を焼き直す:

            - chat_messages       → chat.message   （システムメッセージ除外）
            - scenario_turns      → scene.turn     （参加キャラ1人につき封筒1行）
            - inscribed_memories  → memory.inscribed（全行）＋ memory.forgotten（soft delete 行）
            - tool_call_events    → memory.carved / memory.recalled（status=ok のみ）

        occurred_at は各源の created_at（forgotten は deleted_at）。
        実行済みかは global_settings の marker キーで判定する（冪等）。
        バックフィル中に解決できないキャラクター（削除済み・名前変更済み）の行は
        封筒を作らずスキップする（正本は「現存するキャラのタイムライン」）。
        """
        import json
        import uuid as uuid_mod
        from datetime import datetime as dt

        MARKER_KEY = "timeline_backfill_done"

        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "timeline_events" not in tables or "global_settings" not in tables:
                return
            done = conn.exec_driver_sql(
                "SELECT value FROM global_settings WHERE key = ?", (MARKER_KEY,)
            ).fetchone()
            if done is not None:
                return  # 実行済み

            now_str = dt.now().isoformat(sep=" ")

            def _insert_envelope(
                character_id: str,
                event_type: str,
                occurred_at,
                *,
                actor: str | None = None,
                counterpart: str | None = None,
                origin: str = "real",
                modality: str | None = None,
                session_id: str | None = None,
                source_table: str | None = None,
                source_id: str | None = None,
                payload: dict | None = None,
            ) -> None:
                """封筒1行を raw SQL で挿入する（バックフィル専用の内部ヘルパ）。"""
                conn.exec_driver_sql(
                    "INSERT INTO timeline_events "
                    "(id, character_id, event_type, occurred_at, actor, counterpart, "
                    " origin, modality, session_id, source_table, source_id, "
                    " intent_id, payload, retracted_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?)",
                    (
                        str(uuid_mod.uuid4()), character_id, event_type,
                        occurred_at or now_str, actor, counterpart,
                        origin, modality, session_id, source_table, source_id,
                        json.dumps(payload, ensure_ascii=False) if payload else None,
                        now_str,
                    ),
                )

            # キャラクター名 → ID の解決表（model_id / target からの逆引き用）
            name_to_id: dict[str, str] = {}
            if "characters" in tables:
                for cid, cname in conn.exec_driver_sql(
                    "SELECT id, name FROM characters"
                ).fetchall():
                    if cname and cname not in name_to_id:
                        name_to_id[str(cname)] = str(cid)

            # 1. chat_messages → chat.message
            if "chat_messages" in tables and "chat_sessions" in tables:
                rows = conn.exec_driver_sql(
                    "SELECT m.id, m.session_id, m.role, m.face_to_face, m.created_at, "
                    "       s.model_id "
                    "FROM chat_messages m JOIN chat_sessions s ON m.session_id = s.id "
                    "WHERE (m.is_system_message IS NULL OR m.is_system_message = 0)"
                ).fetchall()
                for mid, sid, role, face, created, model_id in rows:
                    char_name = str(model_id or "").rsplit("@", 1)[0]
                    char_id = name_to_id.get(char_name)
                    if not char_id:
                        continue
                    _insert_envelope(
                        char_id, "chat.message", created,
                        actor="user" if role == "user" else "character",
                        counterpart="user",
                        origin="real",
                        modality="face" if face else "text",
                        session_id=str(sid),
                        source_table="chat_messages",
                        source_id=str(mid),
                    )

            # 2. scenario_turns → scene.turn（参加キャラ1人につき封筒1行）
            if (
                "scenario_turns" in tables
                and "scenario_sessions" in tables
                and "scenarios" in tables
            ):
                rows = conn.exec_driver_sql(
                    "SELECT t.id, t.session_id, t.speaker_type, t.speaker_id, "
                    "       t.speaker_name, t.created_at, "
                    "       ss.engine_type, ss.pc_assignments, sc.owner_character_id "
                    "FROM scenario_turns t "
                    "JOIN scenario_sessions ss ON t.session_id = ss.id "
                    "JOIN scenarios sc ON ss.scenario_id = sc.id"
                ).fetchall()
                for (tid, sid, sp_type, sp_id, sp_name, created,
                     engine_type, asn_raw, owner_id) in rows:
                    is_usual = engine_type == "usual_days"
                    origin = "usual" if is_usual else "interlude"
                    participant_ids: list[str] = []
                    if is_usual:
                        if owner_id:
                            participant_ids = [str(owner_id)]
                    else:
                        try:
                            asn = json.loads(asn_raw) if asn_raw else []
                        except (json.JSONDecodeError, TypeError):
                            asn = []
                        for a in asn if isinstance(asn, list) else []:
                            if (
                                isinstance(a, dict)
                                and a.get("player_type") == "character"
                                and a.get("character_id")
                            ):
                                participant_ids.append(str(a["character_id"]))
                    for char_id in participant_ids:
                        # actor はタイムライン所有者から見た話者
                        # （create_scenario_turn の dual-write と同じ規則）
                        if sp_type == "character" and str(sp_id or "") == char_id:
                            actor = "character"
                        elif sp_type == "user":
                            actor = "user"
                        elif sp_type == "narrator":
                            actor = "narrator"
                        else:
                            actor = f"npc:{sp_name}"
                        _insert_envelope(
                            char_id, "scene.turn", created,
                            actor=actor,
                            origin=origin,
                            session_id=str(sid),
                            source_table="scenario_turns",
                            source_id=str(tid),
                        )

            # 3. inscribed_memories → memory.inscribed ＋ memory.forgotten
            if "inscribed_memories" in tables:
                rows = conn.exec_driver_sql(
                    "SELECT id, character_id, origin, created_at, deleted_at "
                    "FROM inscribed_memories"
                ).fetchall()
                for mid, char_id, origin, created, deleted in rows:
                    _insert_envelope(
                        str(char_id), "memory.inscribed", created,
                        actor="character",
                        origin=str(origin or "real"),
                        source_table="inscribed_memories",
                        source_id=str(mid),
                    )
                    if deleted:
                        _insert_envelope(
                            str(char_id), "memory.forgotten", deleted,
                            actor="character",
                            origin=str(origin or "real"),
                            source_table="inscribed_memories",
                            source_id=str(mid),
                        )

            # 4. tool_call_events → memory.carved / memory.recalled
            if "tool_call_events" in tables:
                rows = conn.exec_driver_sql(
                    "SELECT id, created_at, target, feature, tool_name, arguments_json "
                    "FROM tool_call_events "
                    "WHERE tool_name IN ('carve_narrative', 'power_recall') "
                    "AND status = 'ok'"
                ).fetchall()
                for eid, created, target, feature, tool_name, args_raw in rows:
                    char_id = name_to_id.get(str(target or ""))
                    if not char_id:
                        continue
                    try:
                        args = json.loads(args_raw) if args_raw else {}
                        if not isinstance(args, dict):
                            args = {}
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    f = str(feature or "").lower()
                    if f.startswith("usual"):
                        origin = "usual"
                    elif f.startswith("scenario"):
                        origin = "interlude"
                    else:
                        origin = "real"
                    if tool_name == "carve_narrative":
                        _insert_envelope(
                            char_id, "memory.carved", created,
                            actor="character",
                            origin=origin,
                            payload={
                                "mode": str(args.get("mode") or "append"),
                                "content": str(args.get("content") or ""),
                            },
                        )
                    else:  # power_recall
                        _insert_envelope(
                            char_id, "memory.recalled", created,
                            actor="character",
                            origin=origin,
                            source_table="tool_call_events",
                            source_id=str(eid),
                        )

            # 実行済みマーカーを立てる（同一トランザクション内。冪等性の担保）
            conn.exec_driver_sql(
                "INSERT INTO global_settings (key, value) VALUES (?, ?)",
                (MARKER_KEY, dt.now().isoformat()),
            )

    def _migrate_add_face_to_face_columns(self) -> None:
        """対面モード（Face-to-Face）関連カラムを追加する。

        - `characters.face_to_face_mode` (INTEGER, NOT NULL DEFAULT 0): 0=テキスト / 1=対面。
          キャラスコープで保持し、1on1チャット画面のトグルで切り替える。うつつスケジューラは
          1 のキャラのスロットをスキップする。
        - `characters.face_to_face_bg_image` (TEXT, NULL可): 対面時の ChatView 背景画像
          （base64 data URI）。
        - `chat_messages.face_to_face` (INTEGER, NOT NULL DEFAULT 0): 当該メッセージが
          交わされた時点のモード。後からハレ履歴をうつつ PC へ流し込む際にラベルを
          切り替えるために使う。

        新規DBは ORM 定義で既に作成されるため何もしない。冪等。
        """
        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "characters" in tables:
                cols = {
                    r[1]
                    for r in conn.exec_driver_sql(
                        "PRAGMA table_info(characters)"
                    ).fetchall()
                }
                if "face_to_face_mode" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE characters "
                        "ADD COLUMN face_to_face_mode INTEGER NOT NULL DEFAULT 0"
                    )
                if "face_to_face_bg_image" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE characters ADD COLUMN face_to_face_bg_image TEXT"
                    )
            if "chat_messages" in tables:
                cols = {
                    r[1]
                    for r in conn.exec_driver_sql(
                        "PRAGMA table_info(chat_messages)"
                    ).fetchall()
                }
                if "face_to_face" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE chat_messages "
                        "ADD COLUMN face_to_face INTEGER NOT NULL DEFAULT 0"
                    )

