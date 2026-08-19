/**
 * 添付表示 — 画像のサムネイルグリッド／フルサイズモーダルと、音声プレイヤー。
 */
import { useState } from "react";
import type { Attachment } from "../../api";
import { attachmentKind } from "../../lib/attachments";

/** 添付の配信URL。実体は uploads_dir に置かれ、Content-Type は DB の mime_type。 */
function attachmentUrl(id: string): string {
  return `/api/chat/attachments/${id}`;
}

/**
 * 添付リストを表示するコンポーネント。
 *
 * 画像はサムネイルを並べ、音声はプレイヤーを1件ずつ縦に積む
 * （音声は `<img>` に食わせると壊れるので、種別で描き分ける）。
 */
export function AttachmentGrid({ attachments }: { attachments: Attachment[] }) {
  const [modalSrc, setModalSrc] = useState<string | null>(null);

  // mime を引けなかった添付は画像扱い（旧レコード互換）。
  const audios = attachments.filter((a) => attachmentKind(a.mime_type) === "audio");
  const images = attachments.filter((a) => attachmentKind(a.mime_type) !== "audio");

  return (
    <>
      {audios.length > 0 && (
        <div className="flex flex-col gap-1.5 items-end mb-1">
          {audios.map((att) => (
            <audio
              key={att.id}
              controls
              preload="none"
              src={attachmentUrl(att.id)}
              className="max-w-full h-9"
            />
          ))}
        </div>
      )}
      {images.length > 0 && (
        <div className="flex flex-wrap gap-1.5 justify-end mb-1">
          {images.map((att) => (
            <button
              key={att.id}
              type="button"
              onClick={() => setModalSrc(attachmentUrl(att.id))}
              className="block rounded-lg overflow-hidden transition-opacity hover:opacity-80"
              style={{ border: "1px solid var(--ch-sep)" }}
            >
              <img
                src={attachmentUrl(att.id)}
                alt="添付画像"
                className="w-20 h-20 object-cover"
              />
            </button>
          ))}
        </div>
      )}
      {modalSrc && (
        <ImageModal src={modalSrc} onClose={() => setModalSrc(null)} />
      )}
    </>
  );
}

/** 画像フルサイズ表示モーダル。 */
export function ImageModal({ src, onClose }: { src: string; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center"
      style={{ backdropFilter: "blur(12px)" }}
      onClick={onClose}
    >
      <button
        onClick={onClose}
        className="absolute top-5 right-5 text-ch-t2 hover:text-ch-t1 text-xl leading-none"
      >
        ✕
      </button>
      <img
        src={src}
        alt="フルサイズ画像"
        className="max-w-[90vw] max-h-[90vh] object-contain rounded-lg"
        style={{ boxShadow: "0 0 60px rgba(0,0,0,0.8)" }}
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
}
