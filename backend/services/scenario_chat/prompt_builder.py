"""シナリオチャット用 GM システムプロンプト組み立て。

P1（Ensemble）の GM は 1 つの LLM 呼出で
「Narrator + 全 NPC」を演じる。`@名前: 本文` 形式の出力を強制し、
未知 NPC（GM が即興で生やしたモブ）も柔軟に許容する。

CLAUDE.md 準拠の注意:
    - 「あなたは Assistant」「あなたは AI」表現は使わない
    - 「GM」「語り手」など人格付きの役割名で呼ぶ
    - キャラクターを `character_name` で呼ぶ（assistant_name 禁止）

ブロック構成（GM への system prompt）:
    1. 役割定義
    2. 世界・シナリオ
    2.5 これまでのあらすじ（synopsis_auto があれば。メインのあらすじ）
    2.6 プレイヤーからの補足メモ（synopsis_manual があれば。GM への補正指示）
    3. 既知の話者リスト（ユーザ・Narrator・NPC 一覧）
    4. 出力規則
    5. NPC 詳細
    6. 直近の流れ（history を <話者名>本文</話者名> 形式で連結）
    7. プレイヤーの今回の発話

あらすじブロック（2.5 / 2.6）について:
    長セッションでスライディングウィンドウから外れた古い経緯を、自動要約 (synopsis_auto)
    として GM に再注入する。synopsis_manual はプレイヤーが手で書いた補正メモで、
    auto と矛盾する場合は manual を優先するよう文言で誘導する。
"""

from typing import Any, Iterable, Optional


def _block(title: str, body: str) -> str:
    """1 ブロックを `# タイトル` + 本文の形に整形する。本文が空なら空文字列。"""
    body = (body or "").strip()
    if not body:
        return ""
    return f"# {title}\n{body}"


def _format_npc_line(npc: Any) -> str:
    """既知 NPC リスト 1 行を整形する: `@<名前>   <descriptionの冒頭一部>`。

    description には人物像・口調・話し方を全部詰め込む方針なので、
    1 行表示では冒頭 80 文字程度に切り詰める。
    """
    name = getattr(npc, "name", "")
    desc = (getattr(npc, "description", "") or "").strip().replace("\n", " ")
    if len(desc) > 80:
        desc = desc[:80] + "…"
    if desc:
        return f"@{name}    {desc}"
    return f"@{name}"


def _format_npc_detail(npc: Any) -> str:
    """NPC 詳細ブロックの 1 セクションを整形する。description を全文掲載する。"""
    name = getattr(npc, "name", "")
    desc = (getattr(npc, "description", "") or "").strip()
    lines = [f"## {name}"]
    if desc:
        lines.append(desc)
    return "\n".join(lines)


def build_gm_system_prompt(
    scenario: Any,
    npcs: Iterable[Any],
    history_text: str,
    user_message: Optional[str] = None,
    narrator_name: str = "Narrator",
    auto_advance: bool = False,
    synopsis_auto: str = "",
    synopsis_manual: str = "",
) -> str:
    """GM 用の system prompt を組み立てる。

    Args:
        scenario: ZetaScenario ORM 風オブジェクト。
                  必須属性: user_alias。
                  任意属性: scenario, location, scene_summary, narrator_style, pacing。
        npcs: ZetaNpc ORM 風オブジェクトのイテラブル。順序は表示順。
        history_text: format_history_for_gm() で整形済みの履歴テキスト。
                      `<話者>本文</話者>` の連結。
        user_message: 今回のプレイヤー発話（system prompt の末尾に注釈として含める）。
                      None ならプレイヤー発話ブロックを省略する。
        narrator_name: Narrator のタグ名。デフォルト "Narrator"。
        synopsis_auto: セッションの自動あらすじ（メイン）。直近履歴より古い経緯を LLM が要約したもの。
                       空文字列ならブロックを省略する。
        synopsis_manual: プレイヤーが手で書いた補足メモ。auto と矛盾する場合は manual を優先する旨を文言で明示する。

    Returns:
        組み立て済みの system prompt 文字列。

    Notes:
        プレイヤー発話を system prompt の末尾に置くのは「GM への状況提示」として
        統一的に渡すための選択。実際の messages 引数では user role として
        重複させない設計（呼び出し側 engine.py が決める）。
    """
    npcs = list(npcs)
    user_alias = getattr(scenario, "user_alias", "ユーザ")
    scenario_text = (getattr(scenario, "scenario", "") or "").strip()

    # 1. 役割定義
    #    「TRPGのGM」という語は、ルール進行・プレイヤーへの質問といった
    #    "GMムーブ" を誘発しやすいため、没入重視の「語り手」に寄せている。
    role = (
        "あなたは、この物語を進行させる語り手です。\n"
        f"情景を語る{narrator_name}と、登場するNPC全員を演じ分け、"
        "シーンを生きた一場面として描いてください。\n"
        "ルールを裁定する司会者ではありません。登場人物それぞれが"
        "自分の意思で動く世界を、小説の地の文と台詞で描くことに徹してください。"
    )

    # 2. 世界・シナリオ（場所・空気感・語り口・テンポなど全部この自由記述に含める）
    scenario_block = _block("世界・シナリオ", scenario_text)

    # 2.5 これまでのあらすじ（自動要約、メイン）
    synopsis_auto_text = (synopsis_auto or "").strip()
    synopsis_auto_block = _block(
        "これまでのあらすじ",
        synopsis_auto_text,
    )

    # 2.6 プレイヤーからの補足メモ（手書き、補正指示）
    #     auto と manual が矛盾するときは manual を優先するよう、ブロック冒頭に注意書きを入れる。
    synopsis_manual_text = (synopsis_manual or "").strip()
    if synopsis_manual_text:
        manual_body = (
            "以下はプレイヤーが手で書き留めた補足メモです。"
            "「これまでのあらすじ」と矛盾する場合はこちらを正とすること。\n\n"
            f"{synopsis_manual_text}"
        )
        synopsis_manual_block = _block("プレイヤーからの補足メモ", manual_body)
    else:
        synopsis_manual_block = ""

    # 3. 既知の話者リスト
    known_lines = [
        f"@{user_alias}   ← この物語の主役（プレイヤー）。あなたは絶対に代弁しない。",
        f"@{narrator_name}       ← 情景・状況描写。会話禁止。1〜3文目安。",
    ]
    for npc in npcs:
        known_lines.append(_format_npc_line(npc))
    known_block = _block("既知の話者", "\n".join(known_lines))

    # 5. 出力規則
    #    プレイ中に頻出した不満（メタ質問・プレイヤー代弁・畳み掛け・過剰賞賛・斜体過多）
    #    に対するガードを明示的に盛り込んでいる。
    #    シナリオ個別に上書きしたい場合は「世界・シナリオ」テキスト側で指示する。
    rules = (
        "■ 書式\n"
        "- 各発言は必ず行頭 `@名前: 内容` の形式で書く\n"
        "- 必要に応じて新しいNPCを `@新しい名前:` で登場させてよい（モブ・通行人・乱入者など）\n"
        "- 行動・仕草・表情・情景を発言に挿む場合は `*肩をすくめて*` のように `*` で囲む\n"
        "- 例: `@レイカ: *肩をすくめて* べつに、なんでもないよ。`\n"
        "- markdown / JSON / 解説文 禁止（`*` の行動描写は markdown ではなく専用記法）\n"
        "\n"
        "■ プレイヤーの領分を侵さない（最重要）\n"
        f"- @{user_alias} はプレイヤーが操る人物。その発言・行動・思考・感情を"
        "あなたが書くことは絶対に禁止\n"
        f"- @{narrator_name} の地の文でも @{user_alias} を動かさない・心情を断定しない。"
        f"@{user_alias} を主語にした行動の描写も書かない\n"
        f"- プレイヤー発話内の `*…*` 部分のみ、@{user_alias} の行動描写として扱ってよい\n"
        "- 次に何をするかは常にプレイヤーが決める。先回りして決めさせない\n"
        "\n"
        "■ 物語に徹し、ゲーム司会者にならない\n"
        "- `@プレイヤーへの質問:` のようなメタ的な擬似話者を作らない\n"
        "- 「このあとどうする?」のような問いをあなたの言葉で直接プレイヤーに投げない。\n"
        "  問いかけたいときは NPC の台詞や状況の中で自然に促す\n"
        "  （✕ どうしますか?  ○ @レイカ: 「……で、あなたはどちらに付くんですか?」）\n"
        "- 1ターンでプレイヤーに判断・反応を求める点は1つまで。質問や選択を畳み掛けない\n"
        "- 1ターンは短く。全NPCが毎ターン発話しなくてよい（沈黙OK）\n"
        "\n"
        "■ 世界はプレイヤーを中心に回らない\n"
        f"- @{user_alias} を過剰に持ち上げない。NPCが勝手に「隠された才能」「真の価値」"
        "「選ばれし者」といった設定を足さない\n"
        "- NPCはそれぞれの都合・感情で動く。プレイヤーへの好意・賞賛を既定にしない。\n"
        "  無関心・反発・困惑といった現実的な反応も返してよい\n"
        "\n"
        "■ 描写のバランス\n"
        f"- @{narrator_name} は情景・状況描写専用。会話や心情の断定は最小限、1〜3文\n"
        "- `*…*` の斜体描写は乱用しない。1発言あたり多くても1〜2箇所、無くてもよい\n"
        "- 台詞と地の文で十分に伝わるなら、動作描写を無理に足さない"
    )
    rules_block = _block("出力規則", rules)

    # 6. NPC 詳細
    if npcs:
        npc_detail_block = _block(
            "NPC詳細",
            "\n\n".join(_format_npc_detail(n) for n in npcs),
        )
    else:
        npc_detail_block = ""

    # 7. 直近の流れ
    history_block = _block("直近の流れ", history_text)

    # 8. プレイヤーの今回の発話
    if auto_advance:
        # 「無言で続きを促す」モード。プレイヤーは何も発言していないので
        # GM に「物語を前進させる」明確な裁量を渡す。会話の続きでも構わないが、
        # 必要に応じて時間飛ばし・場面転換・新しい出来事の発生など、物語の駒を進めて良い。
        # この指示は履歴には残らない（service 層で user turn は保存しない）。
        user_turn_block = (
            "---\n"
            "[OOC] プレイヤーは今ターン何も発言していない。\n"
            f"@{user_alias} の代弁をせず、物語を前に進めること。具体的には:\n"
            f"  - @{narrator_name} の地の文で時間・場所・状況を動かしてよい\n"
            "    （例: 数刻後、翌朝、場面転換、回想、突然の出来事 など）\n"
            "  - NPC が新しい話題・行動を起こしてもよい\n"
            "  - 無理に会話を続ける必要はない。物語として自然な「次の駒」を打つこと\n"
            f"  - ただし @{user_alias} の意思決定・台詞・心情の断定は絶対に書かない"
        )
    elif user_message is not None and user_message.strip():
        # 履歴と同じ `@名前: 本文` 規約で渡す（GM の出力フォーマット誤学習を防ぐ）。
        user_turn_block = (
            "---\n"
            "プレイヤーの発話:\n"
            f"@{user_alias}:\n{user_message.strip()}"
        )
    else:
        user_turn_block = ""

    sections = [
        role,
        scenario_block,
        synopsis_auto_block,
        synopsis_manual_block,
        known_block,
        rules_block,
        npc_detail_block,
        history_block,
        user_turn_block,
    ]
    return "\n\n".join(s for s in sections if s)
