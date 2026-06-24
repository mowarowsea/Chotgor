/**
 * チャット系フックで共有する SSE 受信ユーティリティ。
 *
 * useChat / useScenarioChat はそれぞれイベント種別が大きく異なるため hook 自体は分ける
 * 必要があるが、「SSE for await をエラー込みで消費する」という骨組みは同じ。本ファイルで
 * その骨組みを 1 箇所に集約し、各 hook が onEvent / onCleanup を渡すだけで使える形にする。
 *
 * 設計メモ:
 *   - onEvent は false を返した時点で break する（典型: type === "done" や "error" で終了したい時）。
 *   - エラー時は onError を呼び、catch ブロックの reset 処理は onCleanup で書く。
 *     finally 句を hook ごとに繰り返さずに済む。
 */

/** consumeStream の引数。 */
export interface ConsumeStreamArgs<TEvent> {
  /** 受信元の SSE ジェネレータ（例: streamMessage(...) / streamScenarioMessage(...)）。 */
  stream: AsyncGenerator<TEvent>;
  /**
   * 各イベントの処理コールバック。
   *
   * 戻り値が ``false`` だった場合は for-await ループを break する（"done" や "error" で
   * 早期終了したいケース）。``true`` / ``undefined`` を返すとそのまま継続する。
   */
  onEvent: (ev: TEvent) => boolean | void;
  /**
   * SSE 受信中に例外が起きた場合の処理。
   * 渡されない場合は例外は外へ re-throw される。
   */
  onError?: (e: unknown) => void;
  /**
   * try/catch/finally の finally 部分。エラーの有無にかかわらず最後に呼ばれる。
   * pending bubble / streaming state の最終クリアなどに使う。
   */
  onCleanup?: () => void;
}

/**
 * SSE for-await ループを共通エラーハンドリングで回す utility。
 *
 * useChat / useScenarioChat それぞれの「ストリーム受信ループ」の構造を共通化する。
 * 中身のロジック（どのイベントで何をするか）は各 hook 側の onEvent に閉じ込めるため、
 * 共通骨と固有処理を明確に分離できる。
 */
export async function consumeStream<TEvent>(args: ConsumeStreamArgs<TEvent>): Promise<void> {
  const { stream, onEvent, onError, onCleanup } = args;
  try {
    for await (const ev of stream) {
      const ok = onEvent(ev);
      if (ok === false) break;
    }
  } catch (e) {
    if (onError) {
      onError(e);
    } else {
      throw e;
    }
  } finally {
    onCleanup?.();
  }
}
