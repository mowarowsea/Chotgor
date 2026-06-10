/**
 * キャラクターアバターと画像リゾルバ Context。
 * アプリ全体で 1 つのリゾルバを共有し、CharacterAvatar が自動参照する。
 */
import React, { createContext, useContext, useState } from "react";

import { charHue } from "./colors";

/** キャラクター名 → アバター画像URL を解決する関数の型。 */
export type CharImageResolver = (characterName: string) => string | undefined;

/**
 * キャラクター名からアバター画像URLを解決する Context。
 * アプリ全体で 1 つのリゾルバを共有し、CharacterAvatar が自動参照する。
 * これにより各呼び出し側は characterName を渡すだけで設定画像が表示される。
 */
const CharacterImageContext = createContext<CharImageResolver>(() => undefined);

/** アバター画像リゾルバをツリーに供給するプロバイダ。App ルートで 1 度だけ使う。 */
export function CharacterImageProvider({
  resolve,
  children,
}: {
  resolve: CharImageResolver;
  children: React.ReactNode;
}) {
  return (
    <CharacterImageContext.Provider value={resolve}>
      {children}
    </CharacterImageContext.Provider>
  );
}

/**
 * キャラクターアバター。
 *
 * 画像URLは原則 CharacterImageContext から自動解決する（キャラ設定の画像）。
 * imageUrl を明示指定した場合のみそれを優先する（NPC など Context 外の用途向け）。
 * 画像が無い・ロード失敗時はイニシャル＋色相背景にフォールバックする。
 * 配色はキャラクター名から導出した色相（hue）を使う。
 */
export function CharacterAvatar({
  characterName,
  imageUrl,
  size = 28,
  hue,
}: {
  characterName: string;
  /** 画像URLの明示指定（省略時は Context から解決）。 */
  imageUrl?: string;
  /** アバターの直径（px）。デフォルト 28。 */
  size?: number;
  /** 色相（0–359）。省略時はキャラクター名から導出する。 */
  hue?: number;
}) {
  const [imgFailed, setImgFailed] = useState(false);
  const resolve = useContext(CharacterImageContext);
  // imageUrl が明示指定されていればそれを、無ければ Context のリゾルバで解決する。
  const src = imageUrl ?? resolve(characterName);
  const h = hue ?? charHue(characterName);
  const showImg = !!src && !imgFailed;
  return (
    <div
      className="rounded-full flex items-center justify-center font-semibold shrink-0 overflow-hidden"
      style={{
        width: size,
        height: size,
        fontSize: size * 0.38,
        background: showImg ? undefined : `oklch(56% 0.12 ${h} / 0.15)`,
        color: `oklch(44% 0.14 ${h})`,
      }}
    >
      {showImg ? (
        <img
          src={src}
          alt=""
          className="w-full h-full object-cover"
          onError={() => setImgFailed(true)}
        />
      ) : (
        characterName.charAt(0)
      )}
    </div>
  );
}
