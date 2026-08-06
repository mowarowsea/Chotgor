"""シナリオチャット 「あらすじ」自動蒸留モジュールのテスト。

backend.services.scenario_chat.synopsis の関数群を検証する。
update_auto_synopsis 自体は LLM プロバイダー呼出を含むため、本テストでは
ダミープロバイダで E2E ライクに最小確認のみ行う。

検証する観点:
    - `build_synopsis_system_prompt` の主要要素
      （主役の埋め込み・既存 auto の埋め込み・空のときセクション省略・蒸留方針の明示）
    - `update_auto_synopsis` が new_turns 空・preset 不在・LLM 出力空 などのエッジで
      None を返すこと
    - `update_auto_synopsis` が成功時、既存 auto への単純追記ではなく
      LLM が再蒸留した「全体置き換え版」を返すこと

あらすじは「追記し続けて肥大化する」問題への対策として、既存あらすじと新ターンを
統合し全体を再蒸留する方式へ変更された。本テストはその「全体置き換え」の
不変条件を検証する。
"""

from dataclasses import dataclass
import pytest

from backend.services.scenario_chat.synopsis import (
    build_synopsis_system_prompt,
    update_auto_synopsis,
)


# ─── ヘルパー ──────────────────────────────────────────────────────────────────


@dataclass
class FakeScenario:
    """Scenario 風のダミーオブジェクト。

    gm_preset_id はテンプレートではなくセッション単位の設定（ScenarioSession.gm_preset_id）
    に移したため、ダミーには含めない。テストでは update_auto_synopsis の引数で指定する。
    """

    user_alias: str = "プレイヤー"
    scenario: str | None = None


@dataclass
class FakeTurn:
    """ScenarioTurn 風のダミーオブジェクト。"""

    speaker_type: str
    speaker_name: str
    content: str
    turn_index: int = 0


@dataclass
class FakePreset:
    """LLMModelPreset 風のダミー。"""

    id: str = "preset-001"
    name: str = "TestPreset"
    provider: str = "fake"
    model_id: str = "fake-model"


class FakeProvider:
    """generate_stream_typed() で固定の (type, chunk) 列を返すスタブ。"""

    def __init__(self, chunks: list[tuple[str, str]], raises: Exception | None = None):
        """
        Args:
            chunks: yield する (type, content) ペアのリスト。
            raises: None でなければ generate_stream_typed の呼び出しで送出する例外。
        """
        self.chunks = chunks
        self.raises = raises
        self.received_system_prompt: str | None = None
        self.received_messages: list[dict] | None = None

    async def generate_stream_typed(self, system_prompt: str, messages: list[dict]):
        """テスト用の固定出力をストリーミング風に返す。"""
        self.received_system_prompt = system_prompt
        self.received_messages = messages
        if self.raises is not None:
            raise self.raises
        for typ, content in self.chunks:
            yield typ, content


# ─── build_synopsis_system_prompt（プロンプト構造） ─────────────────────────


class TestBuildSynopsisSystemPrompt:
    """蒸留 LLM 用 system prompt の構造を検証する。

    全体再蒸留方式では、既存 auto は「書き換え禁止のコンテキスト」ではなく
    「再蒸留の対象」として渡される。本クラスはその方針が
    プロンプトに正しく表現されていることを検証する。
    """

    def test_pc_centric_rule_without_single_protagonist(self):
        """あらすじ蒸留プロンプトが特定の主役名に依存せず PC 中心の記述規則を持つこと。

        旧仕様では「主役は @user（プレイヤー）」と単一主人公を名指ししていたが、
        中の人を区別しない方針に伴い、PC たちを軸にする中立的な規則へ変更した。
        """
        out = build_synopsis_system_prompt(
            FakeScenario(), "", user_speaker_name="勇者",
        )
        assert "PC" in out or "プレイヤーキャラクター" in out
        # 単一主人公フレーミングは廃止
        assert "主役" not in out

    def test_existing_auto_embedded(self):
        """既存 auto が「これまでのあらすじ」セクションとして埋め込まれること。"""
        out = build_synopsis_system_prompt(
            FakeScenario(), existing_auto="勇者はレイカと出会った。"
        )
        assert "勇者はレイカと出会った。" in out

    def test_existing_auto_marked_as_redistill_target(self):
        """既存 auto が「再蒸留の対象」として明示されること（追記ではなく蒸留）。"""
        out = build_synopsis_system_prompt(
            FakeScenario(), existing_auto="既存のあらすじ。"
        )
        # 「丸ごとコピーして末尾に足す」ことを禁じ、全体蒸留を求める文言があること
        assert "再蒸留" in out or "蒸留し直す" in out

    def test_empty_existing_omits_section(self):
        """既存 auto が空なら、「これまでのあらすじ」見出しセクションは出ないこと。"""
        out = build_synopsis_system_prompt(FakeScenario(), existing_auto="")
        assert "# これまでのあらすじ" not in out

    def test_scenario_text_included_when_present(self):
        """シナリオ世界観テキストがあれば含まれること。"""
        out = build_synopsis_system_prompt(
            FakeScenario(scenario="ここは雨の街。"), existing_auto=""
        )
        assert "ここは雨の街。" in out

    def test_distill_policy_recent_thick_old_thin(self):
        """「直近は厚く・古い経緯は薄く」の蒸留方針が明示されること。"""
        out = build_synopsis_system_prompt(FakeScenario(), existing_auto="")
        assert "圧縮" in out
        # 古い経緯でも事実関係は残す方針が含まれること
        assert "事実" in out

    def test_existing_auto_adds_inheritance_rules(self):
        """既存 auto があるとき「継承規則」が上乗せされること。

        あらすじは全体置き換えで保存され前世代を残さないため、モデルが
        既存あらすじを無視して新規履歴だけを要約すると物語の前半が永久に失われる
        （本番ログで 2000 字超 → 直近 1 シーンのみ 500 字への転落が複数回発生）。
        その再発防止として、以下 3 点がプロンプトに含まれることを固定する:
            - 与えられる履歴は「続き」であって物語全体ではない
            - 出力は物語の冒頭から書き始める
            - 既存あらすじの区間は圧縮してよいが削除は禁止
        """
        out = build_synopsis_system_prompt(
            FakeScenario(), existing_auto="勇者はレイカと出会った。"
        )
        assert "継承規則" in out
        assert "続き" in out
        assert "冒頭" in out
        assert "削除しない" in out or "削除することでは行わない" in out

    def test_no_existing_auto_omits_inheritance_rules(self):
        """既存 auto が空なら、継承規則も「これまでのあらすじ」への言及も出ないこと。

        初回蒸留・全消し後の再蒸留では引き継ぐ前世代が存在しない。
        存在しないブロックを参照する指示を渡すと、モデルが「前半は与えられている
        はず」と誤認して欠落を捏造で埋める余地が生まれるため、文言ごと落とす。
        """
        out = build_synopsis_system_prompt(FakeScenario(), existing_auto="   \n ")
        assert "継承規則" not in out
        assert "これまでのあらすじ" not in out


# ─── update_auto_synopsis（end-to-end with mock provider） ──────────────────


def _loader_for(preset: FakePreset | None):
    """preset_loader 関数を作る（preset が None なら常に None を返す）。"""
    def loader(preset_id: str):
        return preset
    return loader


def _factory_for(provider: FakeProvider):
    """provider_factory 関数を作る。"""
    def factory(provider_id: str, model: str, settings: dict, **kwargs):
        return provider
    return factory


class TestUpdateAutoSynopsis:
    """`update_auto_synopsis` の振る舞いを mock プロバイダで検証する。

    主観点:
        - new_turns 空 → None
        - preset_loader が None → None
        - LLM 出力が空 → None
        - LLM 呼出例外 → None（既存を破壊しない）
        - 通常成功時、既存 auto への追記ではなく LLM 再蒸留結果を全体置き換えで返すこと
        - 既存 auto が蒸留用 system prompt に含まれること
    """

    @pytest.mark.asyncio
    async def test_empty_new_turns_returns_none(self):
        """new_turns が空なら何もせず None を返すこと。"""
        provider = FakeProvider(chunks=[("text", "蒸留結果")])
        result = await update_auto_synopsis(
            scenario=FakeScenario(),
            new_turns=[],
            existing_auto="既存",
            settings={},
            preset_loader=_loader_for(FakePreset()),
            synopsis_preset_id="preset-001",
            provider_factory=_factory_for(provider),
        )
        assert result is None
        # プロバイダは呼ばれていない
        assert provider.received_system_prompt is None

    @pytest.mark.asyncio
    async def test_missing_preset_returns_none(self):
        """preset_loader が None を返したら None。"""
        provider = FakeProvider(chunks=[("text", "蒸留結果")])
        result = await update_auto_synopsis(
            scenario=FakeScenario(),
            new_turns=[
                FakeTurn(speaker_type="user", speaker_name="P", content="やぁ")
            ],
            existing_auto="",
            settings={},
            preset_loader=_loader_for(None),
            synopsis_preset_id="preset-001",
            provider_factory=_factory_for(provider),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_llm_output_returns_none(self):
        """LLM 出力が空文字列のみなら None を返すこと（既存を破壊しない）。"""
        provider = FakeProvider(chunks=[("text", "   "), ("text", "")])
        result = await update_auto_synopsis(
            scenario=FakeScenario(),
            new_turns=[
                FakeTurn(speaker_type="user", speaker_name="P", content="やぁ")
            ],
            existing_auto="既存テキスト",
            settings={},
            preset_loader=_loader_for(FakePreset()),
            synopsis_preset_id="preset-001",
            provider_factory=_factory_for(provider),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_distill_replaces_whole(self):
        """通常成功時、LLM が再蒸留した全体版で置き換わること（既存への追記ではない）。"""
        provider = FakeProvider(chunks=[("text", "蒸留し直した物語全体のあらすじ。")])
        result = await update_auto_synopsis(
            scenario=FakeScenario(),
            new_turns=[
                FakeTurn(speaker_type="user", speaker_name="P", content="A"),
                FakeTurn(speaker_type="narrator", speaker_name="N", content="B"),
            ],
            existing_auto="昔の出来事。",
            settings={},
            preset_loader=_loader_for(FakePreset()),
            synopsis_preset_id="preset-001",
            provider_factory=_factory_for(provider),
        )
        # 既存 auto を末尾連結せず、LLM 出力そのものを全体置き換えで返す
        assert result == "蒸留し直した物語全体のあらすじ。"
        assert "昔の出来事。" not in result

    @pytest.mark.asyncio
    async def test_existing_auto_in_system_prompt(self):
        """既存 auto が蒸留用 system prompt に再蒸留の対象として埋め込まれること。"""
        provider = FakeProvider(chunks=[("text", "x")])
        await update_auto_synopsis(
            scenario=FakeScenario(),
            new_turns=[
                FakeTurn(speaker_type="user", speaker_name="P", content="A")
            ],
            existing_auto="統合対象の既存記述",
            settings={},
            preset_loader=_loader_for(FakePreset()),
            synopsis_preset_id="preset-001",
            provider_factory=_factory_for(provider),
        )
        assert provider.received_system_prompt is not None
        assert "統合対象の既存記述" in provider.received_system_prompt

    @pytest.mark.asyncio
    async def test_user_message_repeats_constraint_after_history(self):
        """既存 auto があるとき、履歴の**後ろ**にも継承制約の念押しが置かれること。

        新規履歴は数万字に達することがあり、その場合モデルの注意は履歴本文へ
        強く引かれる。system prompt 側の指示だけでは「履歴＝物語の全体」と
        誤認され、直近の場面だけを要約して既存あらすじを消し飛ばす事故が起きる。
        履歴の末尾（＝モデルが最後に読む位置）に制約を再掲することでこれを抑える。
        """
        provider = FakeProvider(chunks=[("text", "x")])
        await update_auto_synopsis(
            scenario=FakeScenario(),
            new_turns=[
                FakeTurn(speaker_type="user", speaker_name="P", content="A")
            ],
            existing_auto="既存のあらすじ",
            settings={},
            preset_loader=_loader_for(FakePreset()),
            synopsis_preset_id="preset-001",
            provider_factory=_factory_for(provider),
        )
        assert provider.received_messages is not None
        content = provider.received_messages[0]["content"]
        # 履歴本文より後ろに制約が来ていること（位置関係まで含めて検証）
        pos_history = content.find("A")
        pos_tail = content.find("これは物語の**続き**であり全体ではありません")
        assert pos_tail > pos_history > 0

    @pytest.mark.asyncio
    async def test_user_message_without_existing_has_no_tail(self):
        """既存 auto が無いときは、存在しない前世代を参照する念押しを付けないこと。"""
        provider = FakeProvider(chunks=[("text", "x")])
        await update_auto_synopsis(
            scenario=FakeScenario(),
            new_turns=[
                FakeTurn(speaker_type="user", speaker_name="P", content="A")
            ],
            existing_auto="",
            settings={},
            preset_loader=_loader_for(FakePreset()),
            synopsis_preset_id="preset-001",
            provider_factory=_factory_for(provider),
        )
        assert provider.received_messages is not None
        content = provider.received_messages[0]["content"]
        assert "これまでのあらすじ" not in content

    @pytest.mark.asyncio
    async def test_provider_exception_returns_none(self):
        """LLM 呼出で例外が起きたら None を返し既存を破壊しないこと。"""
        provider = FakeProvider(chunks=[], raises=RuntimeError("LLM 故障"))
        result = await update_auto_synopsis(
            scenario=FakeScenario(),
            new_turns=[
                FakeTurn(speaker_type="user", speaker_name="P", content="A")
            ],
            existing_auto="守りたい既存",
            settings={},
            preset_loader=_loader_for(FakePreset()),
            synopsis_preset_id="preset-001",
            provider_factory=_factory_for(provider),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_provider_error_chunk_returns_none(self):
        """プロバイダ由来エラー chunk("error", ...) を受けたら None を返し既存を破壊しないこと。

        APIキー未設定・SDK 例外・safety filter ブロック等で typed stream に
        ("error", ...) が流れてきたとき、それまでの text chunk が混じっていても
        蒸留結果として採用してはいけない（混入した部分的応答であらすじを上書きすると
        次回再試行できなくなるため）。
        """
        provider = FakeProvider(
            chunks=[
                ("text", "途中まで蒸留した文章"),
                ("error", "[Anthropic API error: 429 rate limit]"),
            ]
        )
        result = await update_auto_synopsis(
            scenario=FakeScenario(),
            new_turns=[
                FakeTurn(speaker_type="user", speaker_name="P", content="A")
            ],
            existing_auto="守りたい既存",
            settings={},
            preset_loader=_loader_for(FakePreset()),
            synopsis_preset_id="preset-001",
            provider_factory=_factory_for(provider),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_provider_error_chunk_first_returns_none(self):
        """ストリーム冒頭で error chunk が来た場合も None を返すこと（APIキー未設定など）。"""
        provider = FakeProvider(
            chunks=[("error", "[Error: anthropic_api_key が設定されていません]")]
        )
        result = await update_auto_synopsis(
            scenario=FakeScenario(),
            new_turns=[
                FakeTurn(speaker_type="user", speaker_name="P", content="A")
            ],
            existing_auto="守りたい既存",
            settings={},
            preset_loader=_loader_for(FakePreset()),
            synopsis_preset_id="preset-001",
            provider_factory=_factory_for(provider),
        )
        assert result is None
