/**
 * 添付ファイルの種別判定 — backend/lib/attachments.py のフロント側対応物。
 *
 * 受け入れ可否の正はあくまで backend（アップロードAPI と送信時ガード）だが、
 * FileDialog の accept と選択直後の検査をここで行い、渡らないものを
 * 「渡ったように見える」状態にしない。
 */

/** 添付種別。プロバイダーの attachment_kinds と同じ語彙。 */
export type AttachmentKind = "image" | "audio" | "text";

/** Gemini が inline_data で受け取れる音声形式（backend の ALLOWED_AUDIO_MIME_TYPES と対）。 */
const ALLOWED_AUDIO_MIME_TYPES = [
  "audio/mpeg",
  "audio/mp3",
  "audio/wav",
  "audio/x-wav",
  "audio/ogg",
  "audio/flac",
  "audio/aac",
];

/** text/* 以外でテキストとして読める MIME（backend の _TEXT_APPLICATION_MIME_TYPES と対）。 */
const TEXT_APPLICATION_MIME_TYPES = [
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
];

/**
 * MIME が当てにならないときにテキストとみなす拡張子（backend の TEXT_EXTENSIONS と対）。
 * Windows のブラウザは .md を空 MIME、.csv を vnd.ms-excel、.ts を video/mp2t で渡してくる。
 */
const TEXT_EXTENSIONS = [
  ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
  ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
  ".xml", ".html", ".htm", ".css", ".svg",
  ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".java", ".kt", ".c", ".h",
  ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".lua", ".sql",
  ".sh", ".bash", ".ps1", ".bat", ".cmd", ".srt", ".vtt",
];

/** テキスト添付の上限（backend の MAX_TEXT_ATTACHMENT_BYTES と対）。 */
const MAX_TEXT_ATTACHMENT_BYTES = 1024 * 1024;

/** `audio/mpeg; codecs=...` のようなパラメータ付き MIME を素の型へ落とす。 */
function normalizeMime(mimeType: string | undefined): string {
  return (mimeType ?? "").split(";")[0].trim().toLowerCase();
}

/** MIME から添付種別を導出する。Chotgor が扱えない形式は null。 */
export function attachmentKind(mimeType: string | undefined): AttachmentKind | null {
  const mime = normalizeMime(mimeType);
  if (mime.startsWith("image/")) return "image";
  if (ALLOWED_AUDIO_MIME_TYPES.includes(mime)) return "audio";
  if (mime.startsWith("text/") || TEXT_APPLICATION_MIME_TYPES.includes(mime)) return "text";
  return null;
}

/**
 * 選択されたファイルの添付種別。MIME から出なければ拡張子でテキストを救済する
 * （backend の resolve_upload_mime と同じ規則）。
 */
export function fileAttachmentKind(file: File): AttachmentKind | null {
  const kind = attachmentKind(file.type);
  if (kind !== null) return kind;
  const dot = file.name.lastIndexOf(".");
  const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
  return TEXT_EXTENSIONS.includes(ext) ? "text" : null;
}

/** 種別リストから FileDialog の accept 属性値を組み立てる。 */
export function acceptAttribute(kinds: AttachmentKind[]): string {
  const patterns: string[] = [];
  if (kinds.includes("image")) patterns.push("image/*");
  if (kinds.includes("audio")) patterns.push(...ALLOWED_AUDIO_MIME_TYPES);
  if (kinds.includes("text")) {
    patterns.push("text/*", ...TEXT_APPLICATION_MIME_TYPES, ...TEXT_EXTENSIONS);
  }
  return patterns.join(",");
}

/** 種別リストから添付ボタンの title（何を渡せるか）を組み立てる。 */
export function attachmentButtonTitle(kinds: AttachmentKind[]): string {
  const labels: string[] = [];
  if (kinds.includes("image")) labels.push("画像");
  if (kinds.includes("audio")) labels.push("音声");
  if (kinds.includes("text")) labels.push("テキスト");
  return labels.length > 0 ? `${labels.join("・")}を添付` : "添付";
}

/**
 * 選択されたファイルが渡せるかを検査し、渡せない理由を返す（渡せるなら null）。
 *
 * スマホの FileDialog は accept を尊重しないことがあるため、選択後にも見る。
 */
export function rejectionReason(file: File, kinds: AttachmentKind[]): string | null {
  const kind = fileAttachmentKind(file);
  if (kind === null) {
    return `「${file.name}」は扱えない形式です`;
  }
  if (!kinds.includes(kind)) {
    if (kind === "audio") {
      return `「${file.name}」は音声です。音声を渡せるのは Gemini（google プロバイダー）だけです`;
    }
    return `「${file.name}」はこのモデルへ渡せません`;
  }
  if (kind === "text" && file.size > MAX_TEXT_ATTACHMENT_BYTES) {
    return `「${file.name}」は大きすぎます（テキストは 1 MB まで）`;
  }
  return null;
}
