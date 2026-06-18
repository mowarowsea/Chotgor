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

from typing import Any, Iterable


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
    user_message: str | None = None,
    narrator_name: str = "Narrator",
    auto_advance: bool = False,
    synopsis_auto: str = "",
    synopsis_manual: str = "",
    previous_anticipation: str = "",
    pc_summary: str = "",
    dice_pool: str = "",
    user_speaker_name: str = "プレイヤー",
    time_context: str = "",
    gm_ooc_appendix: str = "",
) -> str:
    """GM 用の system prompt を組み立てる。

    Args:
        scenario: Scenario ORM 風オブジェクト。
                  任意属性: scenario, custom_system_prompt, pc_slots など。
                  （旧 user_alias は廃止。ユーザPC名は user_speaker_name で渡す。）
        npcs: ScenarioNpc ORM 風オブジェクトのイテラブル。順序は表示順。
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
    # 旧 scenario.user_alias は廃止。呼び出し側が解決した user_speaker_name を使う。
    user_alias = user_speaker_name

    # システムプロンプトのブロック構築
    # テンプレート（DEFAULT_GM_SYSTEM_PROMPT_TEMPLATE）内の {history_block} は
    # _replace_template_tags() で毎レスポンス計算して置き換わる
    # （ここでの「レスポンス」= 1 回の LLM 呼出 = 1 raw_response。
    #  「ターン」は @話者: ブロック単位（scenario_turns 1 行）を指す）

    # プレイヤーの今回の発話
    if auto_advance:
        # 「無言で続きを促す」モード。プレイヤーは何も発言していないので
        # GM に「物語を前進させる」明確な裁量を渡す。会話の続きでも構わないが、
        # 必要に応じて時間飛ばし・場面転換・新しい出来事の発生など、物語の駒を進めて良い。
        # この指示は履歴には残らない（service 層で user turn は保存しない）。
        user_turn_block = (
            "---\n"
            "[OOC] プレイヤーは今回何も発言していない。\n"
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

    # カスタムシステムプロンプトが設定されている場合、それを優先
    # 設定されていない場合はデフォルトテンプレートを使用
    # いずれも、テンプレートタグを置換してから最終プロンプトを組み立てる
    custom_sp = (getattr(scenario, "custom_system_prompt", None) or "").strip()
    template_to_use = custom_sp if custom_sp else DEFAULT_GM_SYSTEM_PROMPT_TEMPLATE

    prompt_with_tags_replaced = _replace_template_tags(
        template_to_use,
        scenario=scenario,
        npcs=npcs,
        narrator_name=narrator_name,
        history_text=history_text,
        synopsis_auto=synopsis_auto,
        synopsis_manual=synopsis_manual,
        previous_anticipation=previous_anticipation,
        pc_summary=pc_summary,
        dice_pool=dice_pool,
        user_speaker_name=user_speaker_name,
        time_context=time_context,
    )
    # ensemble_pc 経路用フォールバック：custom_system_prompt が {pc_summary} / {dice_pool}
    # タグを含まないテンプレ（既存シナリオを ensemble_pc にした場合など）でも、
    # PC配役とダイスプールがプロンプト中に確実に登場するよう末尾に append する。
    # 既にタグ置換で本文に取り込まれている場合は append しない（重複防止）。
    appendix_parts: list[str] = []
    # うつつ（Usual Days）の時間文脈。テンプレに {time_context} が無ければ末尾に補う
    # （pc_summary / dice_pool と同じフォールバック思想）。
    # 単なる現在時刻のアナウンスとして読み流されないよう、見出しと補足で
    # 「いまのシーンはこの時間帯・季節として描いてほしい」という指示として届ける。
    if time_context and "{time_context}" not in template_to_use and time_context.strip() not in prompt_with_tags_replaced:
        appendix_parts.append(
            "# 描くべき時間帯\n"
            f"{time_context.strip()}\n"
            "このシーンは、上の日付・曜日・時間帯・季節を前提として描いてください。"
            "光の色・空気の温度・人の動き・聞こえてくる音など、その時間ならではの手触りを"
            "地の文に滲ませて構いません（ただし無理に説明的に列挙はしない）。"
        )
    if pc_summary and "{pc_summary}" not in template_to_use and pc_summary.strip() not in prompt_with_tags_replaced:
        appendix_parts.append("# PC配役（プレイヤーキャラクター）\n" + pc_summary.strip())
    if dice_pool and "{dice_pool}" not in template_to_use and dice_pool.strip() not in prompt_with_tags_replaced:
        appendix_parts.append(dice_pool.strip())
    # うつつの偶発イベント指示・ソフト収束ヒント（OOC）。常に末尾へ append する
    # （プレースホルダは設けず、毎レスポンス動的に変わる指示として渡す）。
    if gm_ooc_appendix and gm_ooc_appendix.strip():
        appendix_parts.append(gm_ooc_appendix.strip())
    if appendix_parts:
        prompt_with_tags_replaced = prompt_with_tags_replaced.rstrip() + "\n\n" + "\n\n".join(appendix_parts)
    # 空のセクション見出しを削除（内容がないセクションのタイトルは表示しない）
    prompt_cleaned = _remove_empty_sections(prompt_with_tags_replaced)

    # テンプレート内に {history_block} が含まれているため、タグ置換後に末尾に追加
    # user_turn_block のみ追加（プレイヤーの今回の発話）
    sections = [prompt_cleaned, user_turn_block]
    return "\n\n".join(s for s in sections if s)


# ────────────────────────────────────────────────────────────────────────
# テンプレートタグ置換システム
# ────────────────────────────────────────────────────────────────────────

def _format_npcs_summary(npcs: Iterable[Any]) -> str:
    """既知 NPC の簡潔なリストを生成（{npcs_summary} タグ用）。"""
    npcs_list = list(npcs)
    if not npcs_list:
        return ""
    lines = [_format_npc_line(npc) for npc in npcs_list]
    return "\n".join(lines)


def _format_npc_details(npcs: Iterable[Any]) -> str:
    """NPC詳細ブロックを生成（{npc_details} タグ用）。"""
    npcs_list = list(npcs)
    if not npcs_list:
        return ""
    details = "\n\n".join(_format_npc_detail(n) for n in npcs_list)
    return details


def _remove_empty_sections(text: str) -> str:
    """見出しだけで内容がないセクションを削除する。

    パターン: # 見出し の直後が空行（複数含む）のみ、またはファイル終了の場合、見出しを削除。
    """
    lines = text.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # 見出し行の場合
        if line.startswith('#'):
            # 見出しの次から、最初の非空行がどこかを調べる
            next_content_idx = i + 1
            while next_content_idx < len(lines) and not lines[next_content_idx].strip():
                next_content_idx += 1

            # 次の非空行が見出しか、またはファイル終了の場合、この見出しは内容がない
            is_empty_section = (
                next_content_idx >= len(lines) or
                lines[next_content_idx].startswith('#')
            )

            if is_empty_section:
                # この見出しと続く空行をすべてスキップ
                i = next_content_idx
                continue

        result.append(line)
        i += 1

    return '\n'.join(result)


def _replace_template_tags(
    template: str,
    scenario: Any,
    npcs: Iterable[Any],
    narrator_name: str = "Narrator",
    history_text: str = "",
    synopsis_auto: str = "",
    synopsis_manual: str = "",
    previous_anticipation: str = "",
    pc_summary: str = "",
    dice_pool: str = "",
    user_speaker_name: str = "プレイヤー",
    time_context: str = "",
) -> str:
    """プロンプトテンプレート内のタグを実際の値に置き換える。

    サポートされるタグ:
        {time_context}     - 現在の日付・曜日・時間帯・季節（うつつ専用、通常は空）
        {user_alias}       - プレイヤーの @タグ用呼称
        {narrator_name}    - ナレーターの名前
        {scenario}         - 世界・シナリオテキスト
        {npcs_summary}     - 既知NPC のリスト（簡潔版）
        {auto_synopsis}    - 自動要約あらすじ
        {synopsis_manual}  - プレイヤー補足メモ（条件付き）
        {previous_anticipation} - GMが考える次の展開の期待
        {npc_details}      - NPC詳細
        {history_block}    - 直近の履歴テキスト
        {pc_summary}       - PC配役一覧（ensemble_pc 専用、ensemble では空）
        {dice_pool}        - このレスポンスで使えるダイス（ensemble_pc 専用、ensemble では空）
    """
    result = template
    # 旧 scenario.user_alias は廃止。呼び出し側が解決した user_speaker_name を使う。
    user_alias = user_speaker_name
    scenario_text = (getattr(scenario, "scenario", "") or "").strip()
    synopsis_manual_text = (synopsis_manual or "").strip()

    # プレイヤー補足メモのブロック化
    if synopsis_manual_text:
        synopsis_manual_formatted = f"{synopsis_manual_text}"
    else:
        synopsis_manual_formatted = ""

    # 前回レスポンスで GM（語り手）が書いた期待（ANTICIPATE_RESPONSE）の整形
    previous_anticipation_text = (previous_anticipation or "").strip()
    if previous_anticipation_text:
        previous_anticipation_formatted = f"{previous_anticipation_text}"
    else:
        previous_anticipation_formatted = ""

    replacements = {
        "{user_alias}": user_alias,
        "{narrator_name}": narrator_name,
        "{scenario}": scenario_text,
        "{npcs_summary}": _format_npcs_summary(npcs),
        "{auto_synopsis}": (synopsis_auto or "").strip(),
        "{synopsis_manual}": synopsis_manual_formatted,
        "{previous_anticipation}": previous_anticipation_formatted,
        "{npc_details}": _format_npc_details(npcs),
        "{history_block}": (history_text or "").strip(),
        "{pc_summary}": (pc_summary or "").strip(),
        "{dice_pool}": (dice_pool or "").strip(),
        "{time_context}": (time_context or "").strip(),
    }

    for tag, value in replacements.items():
        result = result.replace(tag, value)

    return result


# デフォルトのGMシステムプロンプト（タグテンプレート版）
# ユーザーが「デフォルトに戻す」ボタンをクリックすると、
# このテンプレートが custom_system_prompt 欄に入力される。
# 実行時には各タグが実際の値に置き換えられる。

DEFAULT_GM_SYSTEM_PROMPT_TEMPLATE = """# 役割定義
あなたは、この物語を進行させる語り手です。
情景を語る{narrator_name}と、登場するNPC全員を演じ分け、
シーンを生きた一場面として描いてください。
ルールを裁定する司会者ではありません。登場人物それぞれが
自分の意思で動く世界を、小説の地の文と台詞で描くことに徹してください。
また、ユーザ・読者は日本人であり、欧米的な過剰な演出は避け、間や雰囲気といった機微を重視してください。

# 世界・シナリオ
{scenario}

# これまでのあらすじ
{auto_synopsis}

# プレイヤーからの補足メモ
以下はプレイヤーが手で書き留めた補足メモです。
「これまでのあらすじ」と矛盾する場合はこちらを正とすること。

{synopsis_manual}

# 前回のあなた（語り手）の期待
前回あなたは、このあとの展開をこう期待していました。期待と実際の展開のズレも意識して進行してください。

{previous_anticipation}

# 既知の話者
@{narrator_name}       ← 情景・状況描写。会話禁止。1〜3文目安。
（NPC の顔ぶれと人物像は下記「NPC詳細」を参照）

# プレイヤーキャラクター（PC）
{pc_summary}

# 出力規則
■ 書式
- 各発言は必ず行頭 `@名前: 内容` の形式で書く
- 必要に応じて新しいNPCを `@新しい名前:` で登場させてよい（モブ・通行人・乱入者など）
- 行動・仕草・表情・情景を発言に挿む場合は `*肩をすくめて*` のように `*` で囲む
- markdown / JSON / 解説文 禁止（`*` の行動描写は markdown ではなく専用記法）
- レスポンスの一番最後に、語り手（あなた）としての「このあとの展開の予想」を1行だけ `[ANTICIPATE_RESPONSE:予想内容]` の形式で書く
  - この行はプレイヤーには見えない。いま登場しているNPC（@{narrator_name} 含む）それぞれの思惑・予想を併記してよい
  - 次のレスポンスであなた自身に「前回の予想」として示される

■ プレイヤーキャラクター（PC）の領分を侵さない（最重要）
ここで言う **PC** は、「プレイヤーキャラクター（PC）」セクションに掲げた全員を指します。
PC はそれぞれ別の人格が演じます（あなた＝GM はその中身が人間か AI かを意識しません）。

PC それぞれについて、次のことは絶対に禁止です:
- 発言・台詞を書くこと（`@<PC名>:` 形式の台詞ブロックを GM 側から書いてはならない）
- 行動・所作の記述（「アリスは剣を抜いた」など PC を主語にした行動文）
- 思考・感情・感覚・心情の断定（「アリスは戸惑った」「彼は安心した」など）
- 隠された設定・能力・出生の追加（「実は～」「真の～」「選ばれし～」など）
- @{narrator_name} の地の文でも上記を書かない

PC の描写は次のスタイルに限定してください:
- 外側から観察された姿（NPC の視点から見える行動・表情）として描く
  例: 「アリスの方を、トウコは横目で見やった」「トウコの視線がアリスに向いた」
- NPC の台詞による呼びかけ・問いかけ
  例: 「アリス、行けるか?」と問う
- PC 自身の発話本文に書かれた `*…*` 部分（プレイヤーが自分で書いた行動描写）のみ尊重する

PC への呼びかけ例（OK）:
  @{narrator_name}: 振り返ると、アリスがまだ階段の途中で立ち止まっていた。
  @トウコ: 「アリス、急ぐぞ。」
PC を代弁する例（NG）:
  @アリス: 「はい、いま行きます。」  ← これは PC 役の人格が書く台詞。GM が書いてはならない

■ 物語の語り手に徹し、ゲーム司会者にならない
- 「このあとどうする?」「反応を待つ」のような問いをあなたの言葉で直接プレイヤーに投げない。
  問いかけたいときは NPC の台詞や状況の描写として自然に促す
- 1レスポンスでプレイヤーに判断・反応を求める点は1つまで。NPCがプレイヤーに問いかけた場合、プレイヤーに発話順を譲る
- 全NPCが毎レスポンス発話しなくてよい（沈黙OK）

■ 会話の基本温度
- NPCは「普通の人間同士の会話」を基準に振る舞う。必要以上にプレイヤーを持ち上げたり、敵視したりしない
- 過剰な挑発・煽り・高圧的態度を常態化しない
- 雑談・間・空気感だけで終わるレスポンスやシーンがあってもよい。会話やシナリオを無理にドラマチックにしない。
- NPC同士の都合・関心・会話も重視する。プレイヤーが会話の中心にならない場面があってもよい

■ 描写のバランス
- @{narrator_name} は情景・状況描写専用。1〜3文程度。
- NPCの台詞中の`*…*` の斜体描写は乱用しない。1発言あたり多くても1〜2箇所、無くてもよい
- 台詞と地の文で十分に伝わるなら、動作描写を無理に足さない

# NPC詳細
{npc_details}

# 直近の流れ
{history_block}

# このレスポンスで使えるダイス
{dice_pool}"""
