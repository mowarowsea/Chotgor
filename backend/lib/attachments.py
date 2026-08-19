"""添付ファイルの種別・形式判定。

Chotgor が扱える添付（画像・音声）の唯一の判定根拠。
API 層（受け入れ検証）・services 層（コンテンツパート構築）・providers 層
（能力宣言）の3層から参照されるため、どのレイヤにも属さない lib に置く。

種別は専用カラムを持たず mime_type から導出する（導出できるものを持たない）。
"""

#: Gemini が inline_data で受け取れる音声形式に合わせたホワイトリスト。
#: Chotgor が「扱える添付か」の唯一の根拠。プロバイダー適合（音声を渡せるか）は
#: セッションのプリセット次第なので、送信時に別途判定する。
ALLOWED_AUDIO_MIME_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/flac",
    "audio/aac",
}

#: mime → OpenAI `input_audio` の format 値。ホワイトリスト外は None（＝添付として扱わない）。
_AUDIO_FORMATS = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
    "audio/aac": "aac",
}


def normalize_mime(mime_type) -> str:
    """`audio/mpeg; charset=binary` のようなパラメータ付き MIME を素の型へ落とす。"""
    if not isinstance(mime_type, str):
        return ""
    return mime_type.split(";", 1)[0].strip().lower()


def attachment_kind(mime_type) -> str | None:
    """MIME タイプから添付種別（"image" / "audio"）を導出する。

    Chotgor が扱えない MIME には None を返す（＝アップロードを 400 で弾く根拠）。
    """
    mime = normalize_mime(mime_type)
    if mime.startswith("image/"):
        return "image"
    if mime in ALLOWED_AUDIO_MIME_TYPES:
        return "audio"
    return None


def audio_format(mime_type) -> str | None:
    """音声 MIME を OpenAI `input_audio` の `format` 値へ変換する。"""
    return _AUDIO_FORMATS.get(normalize_mime(mime_type))
