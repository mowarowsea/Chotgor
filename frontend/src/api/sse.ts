/** SSE（Server-Sent Events）共通パーサ。 */

/** SSEレスポンスボディを解析してイベントオブジェクトをyieldする共通ジェネレーター。 */
export async function* parseSSEStream<T>(res: Response): AsyncGenerator<T> {
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          yield JSON.parse(line.slice(6)) as T;
        } catch {
          // 不正なJSONはスキップ
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

