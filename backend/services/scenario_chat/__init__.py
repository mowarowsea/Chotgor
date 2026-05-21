"""シナリオチャットサービスパッケージ。

TRPG 風の「GM (Game Master) 兼 Narrator 兼 全 NPC」を 1 つの LLM 呼出で動かす
Ensemble 形式のセッションエンジンを提供する。

構成:
  - service.py         : API 層から呼ばれる Facade。SSE ストリーム生成・ターン保存
  - context.py         : 履歴切り出しと <話者名>本文</話者名> 形式の整形
  - engine.py          : SceneEngine 抽象 + EnsembleEngine 実装
  - prompt_builder.py  : GM 用 system prompt の組み立て
  - parser.py          : LLM 出力の @名前: ストリーミングパーサ

P1 では engine_type="ensemble" 固定。Polyphony 移行時は PolyphonyEngine を追加する。
"""
