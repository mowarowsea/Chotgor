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

