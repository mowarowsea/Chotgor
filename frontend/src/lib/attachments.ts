/**
 * 添付ファイルの種別判定 — backend/lib/attachments.py のフロント側対応物。
 *
 * 受け入れ可否の正はあくまで backend（アップロードAPI と送信時ガード）だが、
 * FileDialog の accept と選択直後の検査をここで行い、渡らないものを
 * 「渡ったように見える」状態にしない。
 */

/** 添付種別。プロバイダーの attachment_kinds と同じ語彙。 */
export type AttachmentKind = "image" | "audio";

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

/** `audio/mpeg; codecs=...` のようなパラメータ付き MIME を素の型へ落とす。 */
function normalizeMime(mimeType: string | undefined): string {
  return (mimeType ?? "").split(";")[0].trim().toLowerCase();
}

/** MIME から添付種別を導出する。Chotgor が扱えない形式は null。 */
export function attachmentKind(mimeType: string | undefined): AttachmentKind | null {
  const mime = normalizeMime(mimeType);
  if (mime.startsWith("image/")) return "image";
  if (ALLOWED_AUDIO_MIME_TYPES.includes(mime)) return "audio";
  return null;
}

/** 種別リストから FileDialog の accept 属性値を組み立てる。 */
export function acceptAttribute(kinds: AttachmentKind[]): string {
  const patterns: string[] = [];
  if (kinds.includes("image")) patterns.push("image/*");
  if (kinds.includes("audio")) patterns.push(...ALLOWED_AUDIO_MIME_TYPES);
  return patterns.join(",");
}

/** 種別リストから添付ボタンの title（何を渡せるか）を組み立てる。 */
export function attachmentButtonTitle(kinds: AttachmentKind[]): string {
  const labels: string[] = [];
  if (kinds.includes("image")) labels.push("画像");
  if (kinds.includes("audio")) labels.push("音声");
  return labels.length > 0 ? `${labels.join("・")}を添付` : "添付";
}

/**
 * 選択されたファイルが渡せるかを検査し、渡せない理由を返す（渡せるなら null）。
 *
 * スマホの FileDialog は accept を尊重しないことがあるため、選択後にも見る。
 */
export function rejectionReason(file: File, kinds: AttachmentKind[]): string | null {
  const kind = attachmentKind(file.type);
  if (kind === null) {
    return `「${file.name}」は扱えない形式です`;
  }
  if (!kinds.includes(kind)) {
    if (kind === "audio") {
      return `「${file.name}」は音声です。音声を渡せるのは Gemini（google プロバイダー）だけです`;
    }
    return `「${file.name}」はこのモデルへ渡せません`;
  }
  return null;
}
