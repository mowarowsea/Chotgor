"""Claude CLI プロバイダーのマルチモーダル入力テスト。

Chotgor の1on1は添付画像を OpenAI vision 形式（data URL）で messages へ積む
（services/chat/content.py）。Claude CLI 側は ``--input-format stream-json`` の
ときだけ Anthropic ネイティブの image ブロックを受け取れるため、このプロバイダーは
会話テキストと画像を NDJSON 1行にまとめて stdin へ流す。

ここで検証するのは、その変換（何を送り、何を捨てるか）と、組み立てた stdin
ペイロードが実際の CLI 起動経路まで届くこと。
"""

from unittest.mock import MagicMock, patch

from backend.providers.claude_cli_provider import (
    ClaudeCliProvider,
    _build_cli_args,
    _build_stdin_payload,
    _extract_latest_images,
    _format_conversation,
)

# 中身は問わない（base64 文字列がそのまま透過されることの確認用）。
PNG_B64 = "iVBORw0KGgoAAAANSUhEUg=="


def _img(data: str = PNG_B64, mime: str = "image/png") -> dict:
    """content.build_message_content が作る vision 形式の画像パートを模す。"""
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def test_format_conversation_multimodal():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "image_url", "image_url": {"url": "..."}}
            ]
        },
        {
            "role": "assistant",
            "content": "Hi there"
        },
        {
            "role": "user",
            "content": "Next message"
        }
    ]

    formatted = _format_conversation(messages, "Ghost")
    assert "<human>Hello </human>" in formatted
    assert "<Ghost>Hi there</Ghost>" in formatted
    assert formatted.endswith("Next message")


class TestExtractLatestImages:
    """添付画像 → Anthropic image ブロック変換の範囲と足切りを検証する。

    変換の正しさに加えて「どこまで送るか」の線引きを守るためのテスト群。
    対象は末尾メッセージのみで、過去ターンの画像は載せない。全履歴分を毎回積むと
    ターンごとの再送になり、画像トークン（おおよそ 幅×高さ÷750）でサブスク運用の
    レート枠を急速に食い潰すため、「今見せられたもの」だけを渡す設計にしている。

    足切り側（data URL でない・Anthropic 非対応 media_type）は例外にせず黙って捨てる。
    画像1枚のために発話まるごと失敗させるより、テキストだけでもキャラクターに
    届いた方がよいという判断。
    """

    def test_converts_latest_turn_image(self):
        """末尾ターンの data URL が base64 の image ブロックへ変換されること。"""
        messages = [{"role": "user", "content": [{"type": "text", "text": "これ見て"}, _img()]}]
        blocks = _extract_latest_images(messages)
        assert blocks == [{
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
        }]

    def test_ignores_images_in_past_turns(self):
        """過去ターンに付いた画像は拾わない（毎ターン再送を避けるため）。"""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "前に見せたやつ"}, _img()]},
            {"role": "assistant", "content": "見たよ"},
            {"role": "user", "content": "さっきの、どう思った？"},
        ]
        assert _extract_latest_images(messages) == []

    def test_multiple_images_keep_order(self):
        """1ターンに複数枚あるときは添付順を保つこと。"""
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "2枚あるよ"},
            _img(data="AAAA"),
            _img(data="BBBB", mime="image/jpeg"),
        ]}]
        blocks = _extract_latest_images(messages)
        assert [b["source"]["data"] for b in blocks] == ["AAAA", "BBBB"]
        assert [b["source"]["media_type"] for b in blocks] == ["image/png", "image/jpeg"]

    def test_plain_text_content_returns_empty(self):
        """content が文字列（画像なしターン）なら空リスト。"""
        assert _extract_latest_images([{"role": "user", "content": "ただの発話"}]) == []

    def test_empty_messages_returns_empty(self):
        assert _extract_latest_images([]) == []

    def test_drops_unsupported_media_type(self):
        """Anthropic が受け付けない media_type は捨てる（送れば API 側で弾かれるため）。"""
        messages = [{"role": "user", "content": [_img(mime="image/bmp")]}]
        assert _extract_latest_images(messages) == []

    def test_drops_non_data_url(self):
        """data URL でない URL 参照は捨てる（base64 を取り出せないため）。"""
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
        ]}]
        assert _extract_latest_images(messages) == []


class TestBuildStdinPayload:
    """stdin へ流す stream-json（NDJSON 1行）の構造を検証する。

    ``--input-format stream-json`` が受け付けるのは user メッセージのみで、
    キャラクター側のターンをイベントとして流し込むことはできない。よって会話履歴は
    従来どおり _format_conversation が作る XML テキスト1本を text ブロックへ入れ、
    その後ろに最新ターンの画像を並べる —— この並び順が崩れると、キャラクターから見て
    「どの発話に添えられた画像か」が分からなくなる。

    画像の有無で入力形式を分岐させないことも併せて守る（テキストだけのターンも
    同じ NDJSON 経路を通る）。
    """

    def _decode(self, payload: bytes) -> dict:
        import json

        text = payload.decode("utf-8")
        assert text.endswith("\n")
        assert text.count("\n") == 1, "stream-json 入力は1行でなければならない"
        return json.loads(text)

    def test_text_block_first_then_images(self):
        blocks = _extract_latest_images(
            [{"role": "user", "content": [_img(data="XXXX")]}]
        )
        event = self._decode(_build_stdin_payload("<history>…</history>\n\nこれ見て", blocks))
        assert event["type"] == "user"
        assert event["message"]["role"] == "user"
        content = event["message"]["content"]
        assert content[0] == {"type": "text", "text": "<history>…</history>\n\nこれ見て"}
        assert content[1]["type"] == "image"
        assert content[1]["source"]["data"] == "XXXX"

    def test_text_only_turn_uses_same_format(self):
        """画像なしでも stream-json 1行を返す（入力経路を分岐させない）。"""
        event = self._decode(_build_stdin_payload("おはよう", []))
        assert event["message"]["content"] == [{"type": "text", "text": "おはよう"}]

    def test_japanese_is_not_escaped(self):
        """日本語はユニコードエスケープへ展開せず生のまま流す（stdin バイト数を膨らませない）。"""
        payload = _build_stdin_payload("おはよう", [])
        assert "おはよう".encode("utf-8") in payload


class TestStreamJsonInputFlag:
    """CLI 起動フラグに ``--input-format stream-json`` が常時載ることを守る。

    このフラグが落ちると CLI は stdin をプレーンテキストとして読み、NDJSON が
    そのまま発話としてキャラクターへ渡ってしまう（画像は届かず、JSON の生文字列を
    見せることになる）。画像有無で分岐しない設計なので、既定フラグとして固定する。
    """

    def test_input_format_flag_present(self):
        args = _build_cli_args("sys prompt")
        assert "--input-format" in args
        assert args[args.index("--input-format") + 1] == "stream-json"


class TestGenerateRawSendsImages:
    """_run_generate_raw が画像を stdin まで運ぶことを、CLI 起動直前で捕まえて検証する。

    かつてこの経路は画像を検出すると system prompt へ「今は画像が見えない」旨の
    注記を足し、キャラクターに見えないと言わせていた。stream-json 入力の導入で
    その但し書きは不要になったため、注記が復活していないことも併せて確認する
    （復活すると、画像は届いているのに見えないと言う矛盾した応答になる）。
    """

    async def test_image_reaches_stdin_and_no_blind_notice(self):
        import json

        captured: dict = {}

        async def fake_run_claude(sys_path, stdin_bytes, **kwargs):
            with open(sys_path, encoding="utf-8") as f:
                captured["system_prompt"] = f.read()
            captured["stdin"] = stdin_bytes
            return MagicMock(returncode=0, stdout=b"", stderr=b"")

        provider = ClaudeCliProvider(character_name="はる")
        messages = [{"role": "user", "content": [{"type": "text", "text": "これ見て"}, _img()]}]

        with patch("backend.providers.claude_cli_provider._run_claude", side_effect=fake_run_claude), \
             patch.object(provider, "_log_request"), \
             patch.object(provider, "_log_response"):
            await provider._run_generate_raw("あなたは はる。", messages)

        event = json.loads(captured["stdin"].decode("utf-8"))
        content = event["message"]["content"]
        assert content[0]["text"].endswith("これ見て")
        assert content[1]["source"]["data"] == PNG_B64
        # 「画像が見えない」旨の但し書きが system prompt へ混ざっていないこと
        assert "cannot" not in captured["system_prompt"].lower()
        assert captured["system_prompt"] == "あなたは はる。"


class TestAudioPartIsIgnored:
    """音声パートが混ざっても Claude CLI 経路が壊れないことを検証する。

    Anthropic は音声入力を持たない（document ブロックへ audio/mpeg を載せると
    「PDF ではない形式または破損したファイル」として API が拒否する。2026-08-19 実測）。
    入口ガード（プロバイダー能力宣言）で止めるのが本筋だが、万一ここまで届いても
    例外を出さず・不正なブロックを組み立てず、テキストだけで発話が成立すること。
    """

    def test_extract_latest_images_ignores_audio(self):
        """input_audio パートは image ブロックへ変換されないこと。"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "これ聴いて"},
                {"type": "input_audio", "input_audio": {"data": PNG_B64, "format": "mp3"}},
            ],
        }]
        assert _extract_latest_images(messages) == []

    def test_conversation_text_survives_audio_part(self):
        """会話テキストの組み立てが音声パートで壊れないこと。"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "これ聴いて"},
                {"type": "input_audio", "input_audio": {"data": PNG_B64, "format": "mp3"}},
            ],
        }]
        assert _format_conversation(messages, "はる") == "これ聴いて"

    def test_stdin_payload_has_text_only(self):
        """stdin ペイロードにはテキストブロックだけが載ること（音声は捨てられる）。"""
        import json

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "これ聴いて"},
                {"type": "input_audio", "input_audio": {"data": PNG_B64, "format": "mp3"}},
            ],
        }]
        payload = _build_stdin_payload(
            _format_conversation(messages, "はる"), _extract_latest_images(messages)
        )
        content = json.loads(payload.decode("utf-8"))["message"]["content"]
        assert content == [{"type": "text", "text": "これ聴いて"}]
