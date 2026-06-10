/**
 * 添付画像表示 — サムネイルグリッドとフルサイズモーダル。
 */
import { useState } from "react";

/**
 * 添付画像IDリストをサムネイルグリッドで表示するコンポーネント。
 */
export function ImageGrid({ imageIds }: { imageIds: string[] }) {
  const [modalSrc, setModalSrc] = useState<string | null>(null);

  return (
    <>
      <div className="flex flex-wrap gap-1.5 justify-end mb-1">
        {imageIds.map((id) => (
          <button
            key={id}
            type="button"
            onClick={() => setModalSrc(`/api/chat/images/${id}`)}
            className="block rounded-lg overflow-hidden transition-opacity hover:opacity-80"
            style={{ border: "1px solid var(--ch-sep)" }}
          >
            <img
              src={`/api/chat/images/${id}`}
              alt="添付画像"
              className="w-20 h-20 object-cover"
            />
          </button>
        ))}
      </div>
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
