"""添付ファイルの種別・形式判定。

Chotgor が扱える添付（画像・音声・テキスト）の唯一の判定根拠。
API 層（受け入れ検証）・services 層（コンテンツパート構築）・providers 層
（能力宣言）の3層から参照されるため、どのレイヤにも属さない lib に置く。

種別は専用カラムを持たず mime_type から導出する（導出できるものを持たない）。
"""

import os

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

#: text/* 以外でテキストとして読める MIME。
_TEXT_APPLICATION_MIME_TYPES = {
    "application/json",
    "application/xml",
    "application/yaml",
    "application/x-yaml",
    "application/toml",
    "application/javascript",
    "application/x-javascript",
    "application/typescript",
    "application/x-sh",
    "application/sql",
    "application/x-ndjson",
}

#: MIME が当てにならないとき（Windows で .md が空 MIME、.csv が vnd.ms-excel、
#: .ts が video/mp2t 等）にテキストとみなす拡張子。
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".xml", ".html", ".htm", ".css", ".svg",
    ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".java", ".kt", ".c", ".h",
    ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".lua", ".sql",
    ".sh", ".bash", ".ps1", ".bat", ".cmd", ".srt", ".vtt",
}

#: テキスト添付の上限。中身が丸ごとプロンプトに載るため、巨大ファイルの誤投入を防ぐ。
MAX_TEXT_ATTACHMENT_BYTES = 1024 * 1024


def normalize_mime(mime_type) -> str:
    """`audio/mpeg; charset=binary` のようなパラメータ付き MIME を素の型へ落とす。"""
    if not isinstance(mime_type, str):
        return ""
    return mime_type.split(";", 1)[0].strip().lower()


def attachment_kind(mime_type) -> str | None:
    """MIME タイプから添付種別（"image" / "audio" / "text"）を導出する。

    Chotgor が扱えない MIME には None を返す（＝アップロードを 400 で弾く根拠）。
    """
    mime = normalize_mime(mime_type)
    if mime.startswith("image/"):
        return "image"
    if mime in ALLOWED_AUDIO_MIME_TYPES:
        return "audio"
    if mime.startswith("text/") or mime in _TEXT_APPLICATION_MIME_TYPES:
        return "text"
    return None


def resolve_upload_mime(content_type, filename) -> str:
    """アップロード時に保存する MIME を決める。

    MIME から種別が出ればそのまま。出ない場合に限り、拡張子がテキスト系なら
    `text/plain` へ正規化する（ブラウザの MIME が当てにならないため）。
    以後の種別判定は保存済み MIME だけで完結する。
    """
    mime = content_type or ""
    if attachment_kind(mime) is not None:
        return mime
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in TEXT_EXTENSIONS:
        return "text/plain"
    return mime


def decode_text(data: bytes) -> str:
    """テキスト添付の中身を文字列へ。UTF-8（BOM 可）→ CP932 の順に試す。"""
    for encoding in ("utf-8-sig", "cp932"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def audio_format(mime_type) -> str | None:
    """音声 MIME を OpenAI `input_audio` の `format` 値へ変換する。"""
    return _AUDIO_FORMATS.get(normalize_mime(mime_type))
