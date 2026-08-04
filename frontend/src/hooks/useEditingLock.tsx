/**
 * バブルのインライン編集中に、下部の新規メッセージ入力欄をロックするための Context。
 *
 * 背景: 編集フォームは各バブルのローカル state で開くため、下部の `MessageInput` は
 * それを知らない。編集中に誤って送信すると、以降のターンを巻き戻す再ストリームが走り、
 * 編集内容が失われる（バブルが再マウントされて入力が消える）。
 * そこで「編集中のバブル数」だけを共有し、1 つでも開いていれば入力欄を無効化する。
 *
 * 使い方:
 *   - 入力欄とバブル群を含むビュー（ChatView / ScenarioChatView）を
 *     `EditingLockProvider` で包む。
 *   - 編集フォームを持つバブルは `useEditingLock(editing)` を呼ぶ。
 *   - `MessageInput` は `useEditingLocked()` で状態を読む。
 *
 * Provider が無い場合は no-op（常にロック解除）として振る舞う。
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

interface EditingLockValue {
  /** 1 つ以上のバブルが編集中か。 */
  locked: boolean;
  /** 編集開始（カウント +1）。 */
  begin: () => void;
  /** 編集終了（カウント -1）。 */
  end: () => void;
}

const noop = () => {};

const EditingLockContext = createContext<EditingLockValue>({
  locked: false,
  begin: noop,
  end: noop,
});

/** 編集ロックの共有スコープ。入力欄と編集可能なバブル群をまとめて包む。 */
export function EditingLockProvider({ children }: { children: React.ReactNode }) {
  // 複数バブルが同時に編集され得る（1on1 の履歴・シナリオの過去ターン）ため、
  // 真偽値ではなくカウントで持つ。1 つ閉じただけでロックが解けないようにする。
  const [count, setCount] = useState(0);

  // begin/end は effect の依存に載るので、参照を固定して不要な再購読を避ける。
  const begin = useCallback(() => setCount((c) => c + 1), []);
  const end = useCallback(() => setCount((c) => Math.max(0, c - 1)), []);

  const value = useMemo(
    () => ({ locked: count > 0, begin, end }),
    [count, begin, end],
  );

  return (
    <EditingLockContext.Provider value={value}>
      {children}
    </EditingLockContext.Provider>
  );
}

/**
 * `editing` が true の間だけ入力欄をロックする。
 * 編集中にバブルがアンマウントされてもクリーンアップで確実に解除される。
 */
export function useEditingLock(editing: boolean) {
  const { begin, end } = useContext(EditingLockContext);
  useEffect(() => {
    if (!editing) return;
    begin();
    return end;
  }, [editing, begin, end]);
}

/** 現在いずれかのバブルが編集中か（入力欄側が読む）。 */
export function useEditingLocked(): boolean {
  return useContext(EditingLockContext).locked;
}
