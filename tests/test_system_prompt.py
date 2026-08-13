"""backend.services.chat.request_builder モジュールのテスト。

build_system_prompt()（安定ブロックのみ）と build_turn_annotation()（毎ターン
変動ブロック＝ターン注釈）、append_turn_annotation()（注釈のメッセージ付加）を検証する。

プロンプトキャッシュ対応（request_builder.py モジュール docstring「二層に分ける理由」参照）により、想起記憶・時刻・
fetched・WM heat 想起・前回の期待はシステムプロンプトから外れ、ターン注釈として
最新 user メッセージの末尾へ付加される二層構成になった。本テストは
「安定ブロックがシステムプロンプトに残り、変動ブロックが注釈側へ移っている」
という分離そのものも回帰防止の対象とする。
"""

from backend.services.chat.request_builder import (
    append_turn_annotation,
    build_system_prompt,
    build_turn_annotation,
)


# ────────────────────────────────────────────────────────────────────────
# build_system_prompt — 安定ブロック（システムプロンプト）
# ────────────────────────────────────────────────────────────────────────

def test_build_system_prompt_basic():
    """キャラクター設定が含まれ、記憶ブロックのヘッダーが存在すること。"""
    char_prompt = "You are a cat."
    prompt = build_system_prompt(char_prompt)

    assert char_prompt in prompt
    assert "## あなたの記憶について" in prompt


def test_build_system_prompt_includes_chotgor_context():
    """【前提】ブロックは常に含まれること。"""
    prompt = build_system_prompt("You are a cat.")
    assert "【前提】" in prompt
    assert "Chotgor" in prompt


def test_build_system_prompt_chotgor_context_after_block1():
    """【前提】ブロックはキャラクター設定の直前（前）に挿入されること。

    実装では【前提】ブロックをキャラクター設定ブロックの前に置き、
    その後にキャラクター固有の system_prompt_block1 が続く。
    """
    char_prompt = "You are a cat."
    prompt = build_system_prompt(char_prompt)
    assert prompt.index("【前提】") < prompt.index(char_prompt)


def test_build_system_prompt_excludes_variable_blocks():
    """変動ブロック（時刻・想起記憶・前回の期待）がシステムプロンプトに現れないこと。

    プロンプトキャッシュ対応の核心: 毎ターン変動する情報がシステムプロンプトに
    混ざると、それ以降（会話履歴含む）が全ターンでキャッシュミスになる。
    build_system_prompt が変動系の引数を受け取らないこと（TypeError）で分離を保証する。
    """
    import pytest
    with pytest.raises(TypeError):
        build_system_prompt("You are a cat.", current_time_str="2026-03-08T12:00:00")
    with pytest.raises(TypeError):
        build_system_prompt("You are a cat.", recalled_memories=[{"content": "x"}])
    with pytest.raises(TypeError):
        build_system_prompt("You are a cat.", previous_anticipation="予想")


def test_build_system_prompt_with_inner_narrative():
    """inner_narrative が設定されているとき、専用ブロックとして含まれること。"""
    prompt = build_system_prompt(
        "You are a cat.",
        inner_narrative="知的好奇心を大切にしたい。",
    )
    assert "inner_narrative" in prompt
    assert "知的好奇心を大切にしたい。" in prompt


def test_build_system_prompt_with_carve_narrative_guide():
    """use_tools=False のとき CARVE_NARRATIVE タグガイドが含まれること。"""
    prompt = build_system_prompt("You are a cat.", use_tools=False)
    assert "[CARVE_NARRATIVE:" in prompt


def test_build_system_prompt_tools_mode_uses_tool_hint():
    """use_tools=True のとき carve_narrative ツール案内（ツール形式）が含まれること。"""
    prompt = build_system_prompt("You are a cat.", use_tools=True)
    assert "carve_narrative" in prompt


def test_build_system_prompt_anticipate_guide_present_both_modes():
    """ANTICIPATE_RESPONSE の出力ガイドが tool-use / タグ方式の両方で常に含まれること。

    予想タグは全プロバイダー一律でテキストタグとして書かせる方針のため、
    use_tools の真偽にかかわらずガイドが入っていなければならない。
    （チャット経路のデフォルト。単発問い合わせでの抑制は次のテストを参照。）
    """
    prompt_tags = build_system_prompt("You are a cat.", use_tools=False)
    prompt_tools = build_system_prompt("You are a cat.", use_tools=True)
    assert "ANTICIPATE_RESPONSE" in prompt_tags
    assert "ANTICIPATE_RESPONSE" in prompt_tools


def test_build_system_prompt_anticipate_guide_suppressed():
    """include_anticipation_guide=False のとき ANTICIPATE_RESPONSE ガイドが省かれること。

    予想は「次のターンを受け取る相手がいる」チャット前提の機能。
    Chronicle/Forget 等の単発問い合わせ（ask_character 系）では次のターンが
    存在しないため、ガイド自体をプロンプトに入れない。tool-use / タグ方式の
    両分岐で抑制されることを検証する。
    """
    prompt_tags = build_system_prompt(
        "You are a cat.", use_tools=False, include_anticipation_guide=False
    )
    prompt_tools = build_system_prompt(
        "You are a cat.", use_tools=True, include_anticipation_guide=False
    )
    assert "ANTICIPATE_RESPONSE" not in prompt_tags
    assert "ANTICIPATE_RESPONSE" not in prompt_tools


def test_build_system_prompt_memories_guide_points_to_annotation():
    """記憶ガイドが「メッセージ末尾の【このターンの文脈】」を案内していること。

    想起記憶がシステムプロンプトからターン注釈へ移ったため、旧文言
    「すでに上に記されています」のままだとキャラクターが記憶を探す場所を誤る。
    ガイド文言が新しい所在を指していることを回帰防止する。
    """
    prompt = build_system_prompt("You are a cat.")
    assert "【このターンの文脈】" in prompt
    assert "すでに上に記されています" not in prompt


# --- 記憶システム縮退時の運用告知ブロック（memory_degraded） ---

def test_build_system_prompt_memory_degraded_includes_notice():
    """memory_degraded=True のとき、記憶縮退の運用告知ブロックが含まれること。

    告知には「記憶が消えたわけではない」の明示が必須。これが無いと、キャラクターが
    想起ゼロの状態を「忘れてしまった」と誤解し、inner_narrative や関係認識を
    書き換えてしまう二次事故につながるため、文言レベルで回帰を防止する。

    また受け皿の案内は「今夜の棚卸し（Chronicle）」であること。Chronicle は当日会話を
    SQLite から読み返すため embedding 障害中も無傷で機能する、嘘のない受け皿である。
    （「復旧後に改めて刻んで」のような案内は、ツールコールの意図が後続ターンの文脈に
    残らない以上、実行不可能なので書かない。）
    """
    prompt = build_system_prompt("You are a cat.", memory_degraded=True)
    assert "【Chotgor運用担当からのお知らせ】" in prompt
    assert "あなたの記憶が消えたわけではありません" in prompt
    assert "棚卸し（Chronicle）" in prompt


def test_build_system_prompt_memory_degraded_default_off():
    """既定（memory_degraded=False）では運用告知ブロックが含まれないこと。

    通常時のシステムプロンプトに障害告知が混入すると、キャラクターが常に
    「記憶が不安定だ」という前提で振る舞ってしまうため、平常時の非注入を保証する。
    """
    prompt = build_system_prompt("You are a cat.")
    assert "【Chotgor運用担当からのお知らせ】" not in prompt


def test_build_system_prompt_memory_notice_placed_before_guide():
    """運用告知ブロックは inner_narrative の後・Chotgor 操作ガイド（常に末尾）の前に置かれること。

    末尾に近いほどキャラクターへの影響力が強いというテンプレ設計
    （inner_narrative と Chotgor 操作ガイドが最優先地帯）に基づき、
    告知が操作ガイド直前という強い位置に差し込まれることを保証する。
    """
    prompt = build_system_prompt(
        "You are a cat.",
        inner_narrative="私は自由な猫だ",
        memory_degraded=True,
    )
    notice_pos = prompt.index("【Chotgor運用担当からのお知らせ】")
    assert prompt.index("## あなた自身の物語") < notice_pos
    assert notice_pos < prompt.index("## あなたの記憶について")


# ────────────────────────────────────────────────────────────────────────
# build_turn_annotation — 変動ブロック（ターン注釈）
# ────────────────────────────────────────────────────────────────────────

def test_turn_annotation_empty_when_no_material():
    """全素材が空のとき注釈は空文字列になること（ヘッダだけの注釈を出さない）。

    素材ゼロでヘッダ付き注釈が出ると、毎ターン無内容な注釈がメッセージ末尾に
    付いてしまい、トークンの無駄かつキャラクターへのノイズになる。
    """
    assert build_turn_annotation() == ""


def test_turn_annotation_has_header_when_material_exists():
    """素材があるとき、注釈冒頭に Chotgor からの説明ヘッダが付くこと。

    注釈は user メッセージの末尾に付加されるため、「ユーザの発言ではない」ことを
    キャラクター本人に明示しないと、発言の帰属を誤解する恐れがある。
    """
    annotation = build_turn_annotation(
        enable_time_awareness=True,
        current_time_str="2026-03-08T12:00:00",
    )
    assert "【このターンの文脈（Chotgorより）】" in annotation
    assert "ユーザの発言ではありません" in annotation


def test_turn_annotation_time_disabled_by_default():
    """enable_time_awareness=False のとき時刻情報は含まれないこと。"""
    annotation = build_turn_annotation(current_time_str="2026-03-08T12:00:00")
    assert "【現在時刻" not in annotation


def test_turn_annotation_time_requires_current_time_str():
    """enable_time_awareness=True でも current_time_str が空なら時刻ブロックは入らないこと。"""
    annotation = build_turn_annotation(
        enable_time_awareness=True,
        current_time_str="",
    )
    assert "【現在時刻" not in annotation


def test_turn_annotation_time_no_last_interaction():
    """前回の経過時間がない場合は現在時刻のみ含まれること。"""
    annotation = build_turn_annotation(
        enable_time_awareness=True,
        current_time_str="2026-03-08T12:00:00",
    )
    assert "【現在時刻：2026-03-08T12:00:00】" in annotation
    assert "【前回の交流から" not in annotation


def test_turn_annotation_with_time():
    """時刻情報（現在時刻＋前回交流からの経過）が正しく含まれること。"""
    annotation = build_turn_annotation(
        enable_time_awareness=True,
        current_time_str="2026-03-05 12:00",
        time_since_last_interaction="1 hour",
    )
    assert "【現在時刻：2026-03-05 12:00】" in annotation
    assert "【前回の交流から：1 hour】" in annotation


def test_turn_annotation_with_memories():
    """recalled_memories を渡すと記憶内容が見出し付きで含まれること。

    見出し（## Relevant Memories from Past Conversations）はシステムプロンプト
    時代と同一であること — 現れる場所だけが変わり、キャラクターに見える情報の
    中身・形式は変えないという二層分離の設計を文言レベルで保証する。
    """
    memories = [{"content": "User likes fish", "metadata": {"category": "user"}}]
    annotation = build_turn_annotation(recalled_memories=memories)

    assert "## Relevant Memories from Past Conversations" in annotation
    assert "User likes fish" in annotation


def test_turn_annotation_with_identity_memories():
    """identity 枠の記憶が Identity 見出しの下に含まれること。"""
    identity = [{"content": "私は猫である", "metadata": {"category": "identity"}}]
    annotation = build_turn_annotation(recalled_identity_memories=identity)
    assert "### Identity" in annotation
    assert "私は猫である" in annotation


def test_turn_annotation_memories_does_not_use_old_memory_tag():
    """古い [MEMORY:...] タグガイドが注釈に含まれないこと（改名済み確認）。"""
    import re
    annotation = build_turn_annotation(
        recalled_memories=[{"content": "内容", "metadata": {"category": "user"}}],
    )
    old_tag = re.compile(r'\[MEMORY:')
    assert not old_tag.search(annotation), (
        "ターン注釈に古いタグ形式 [MEMORY:...] が残っています"
    )


def test_turn_annotation_with_web():
    """fetched_contents を渡すと Fetched Web Content ブロックが含まれること。"""
    web = [{"url": "http://example.com", "content": "Example page"}]
    annotation = build_turn_annotation(fetched_contents=web)

    assert "## Fetched Web Content" in annotation
    assert "Example page" in annotation


def test_turn_annotation_with_wm_recalled():
    """wm_recalled_threads を渡すと WM heat 想起ブロックが含まれること。"""
    threads = [{"id": "t1", "type": "task", "summary": "課題A", "latest_post": "進捗あり"}]
    annotation = build_turn_annotation(wm_recalled_threads=threads)
    assert "## ワーキングメモリ：いま前景にある課題・話題" in annotation
    assert "課題A" in annotation


def test_turn_annotation_previous_anticipation_injected():
    """previous_anticipation を渡すと「前回のあなたの期待」ブロックが本文込みで挿入されること。"""
    annotation = build_turn_annotation(
        previous_anticipation="次は相手が笑うと予想していた",
    )
    assert "## 前回のあなたの期待（予想）" in annotation
    assert "次は相手が笑うと予想していた" in annotation


def test_turn_annotation_previous_anticipation_absent_when_empty():
    """previous_anticipation が空（デフォルト）のときは予想注入ブロックが挿入されないこと。"""
    annotation = build_turn_annotation(
        enable_time_awareness=True,
        current_time_str="2026-03-08T12:00:00",
    )
    assert "## 前回のあなたの期待（予想）" not in annotation


# ────────────────────────────────────────────────────────────────────────
# append_turn_annotation — 注釈のメッセージ付加
# ────────────────────────────────────────────────────────────────────────

def test_append_annotation_to_last_user_message():
    """注釈が最新 user メッセージの末尾へ追記されること。"""
    messages = [
        {"role": "user", "content": "こんにちは"},
        {"role": "assistant", "content": "やあ"},
        {"role": "user", "content": "元気？"},
    ]
    result = append_turn_annotation(messages, "【注釈】")
    assert result[-1]["content"] == "元気？\n\n【注釈】"
    # 過去のメッセージは無変更
    assert result[0]["content"] == "こんにちは"
    assert result[1]["content"] == "やあ"


def test_append_annotation_does_not_mutate_original():
    """元の messages リスト・dict が変更されないこと（コピーへの付加）。

    注釈が元リストへ混入すると、DB 保存や次ターンの履歴構築に注釈が漏れて
    「履歴プレフィックスを汚さない」というキャッシュ設計の前提が崩れる。
    """
    original = [{"role": "user", "content": "元気？"}]
    result = append_turn_annotation(original, "【注釈】")
    assert original[0]["content"] == "元気？"
    assert result is not original
    assert result[0] is not original[0]


def test_append_annotation_empty_returns_same():
    """注釈が空文字列のとき messages がそのまま返ること（コピーも作らない）。"""
    messages = [{"role": "user", "content": "元気？"}]
    assert append_turn_annotation(messages, "") is messages


def test_append_annotation_multipart_content():
    """画像添付等のマルチパート content にはテキストパートとして末尾追加されること。"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "この画像見て"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
            ],
        },
    ]
    result = append_turn_annotation(messages, "【注釈】")
    parts = result[0]["content"]
    assert parts[-1]["type"] == "text"
    assert "【注釈】" in parts[-1]["text"]
    # 元のマルチパートリストは無変更
    assert len(messages[0]["content"]) == 2


def test_append_annotation_last_not_user_appends_new_turn():
    """最終メッセージが user でない場合、注釈のみの user メッセージが追加されること。

    通常経路では起きない並びだが、注釈（想起記憶等）を黙って落とすと
    キャラクターが文脈を失うため、独立ターンとして届ける安全弁を検証する。
    """
    messages = [{"role": "assistant", "content": "やあ"}]
    result = append_turn_annotation(messages, "【注釈】")
    assert len(result) == 2
    assert result[-1] == {"role": "user", "content": "【注釈】"}


# ────────────────────────────────────────────────────────────────────────
# WM スレッド一覧ブロック — 行の短縮と省略の告知
# （current-spec/memory_recall_algorithm.md §4.3）
# ────────────────────────────────────────────────────────────────────────

_WM_OPEN_THREAD = {
    "id": "wm-open-0001",
    "type": "task",
    "summary": "進行中の課題",
    "atmosphere_tag": "手応えあり",
    "importance": 0.7,
    "is_open": True,
}
_WM_CLOSED_THREAD = {
    "id": "wm-closed-001",
    "type": "topic",
    "summary": "決着した話題",
    "atmosphere_tag": "きれいに終わった",
    "importance": 0.4,
    "is_open": False,
}


def test_wm_all_block_closed_thread_is_headline_only():
    """Close 済みスレッドは見出しだけの短縮行になること。

    一覧は毎ターン全件が載る構造で、Close 済みは増え続ける。決着済みの話に
    温度感（atmosphere_tag）と重要度は要らないため、Open 行では出るこれらが
    Close 行では落ちることを固定する。summary と短縮 ID は Close でも残る
    （read_working_memory_thread / reopen で参照するため）。
    """
    prompt = build_system_prompt(
        "You are a cat.",
        wm_all_threads=[_WM_OPEN_THREAD, _WM_CLOSED_THREAD],
    )
    # Open 行は従来どおり温度感と重要度つき
    assert "手応えあり" in prompt
    assert "重要度0.70" in prompt
    # Close 行は見出しのみ
    assert "決着した話題" in prompt
    assert "wm-close" in prompt
    assert "きれいに終わった" not in prompt
    assert "重要度0.40" not in prompt


def test_wm_all_block_announces_omitted_closed_threads():
    """一覧から省いた Close 済みの本数と、取り出す手段が告知されること。

    Close 済みを直近ぶんに絞ると「過去に越えてきたこと」がプロンプトから
    見えなくなる。視界から黙って消すのではなく、残数とツール名を必ず添える
    （存在は見えていて、中身は必要なときに開く、という形にするため）。
    """
    prompt = build_system_prompt(
        "You are a cat.",
        wm_all_threads=[_WM_OPEN_THREAD],
        wm_omitted_closed=14,
    )
    assert "14 本" in prompt
    assert "read_working_memory_list" in prompt


def test_wm_all_block_no_notice_when_nothing_omitted():
    """省略が無いターンでは告知行を出さないこと（常時出ると雑音になる）。"""
    prompt = build_system_prompt(
        "You are a cat.",
        wm_all_threads=[_WM_OPEN_THREAD],
        wm_omitted_closed=0,
    )
    assert "read_working_memory_list で一覧を取り出せます" not in prompt
