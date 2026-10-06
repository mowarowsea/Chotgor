"""read_dear_ghost（キャラクター向けガイドの読み返し）のテスト。"""

from backend.character_actions.guide_reader import DEAR_GHOST_PATH, read_dear_ghost
from backend.character_actions.tool_specs import BASE_TOOL_SPECS

_SAMPLE = """\
# ガイド

> 最終更新: 2026-10-06

---

## 1. 記憶

章1の前置き。

### 1.1 種類

種類の本文。

```text
## 9. これはコードブロック内なので見出しではない
```

### 1.2 抽出

抽出の本文。

---

## 2. 道具

### 2.1 inscribe

刻む本文。
"""


class TestReadDearGhostSections:
    """section 引数による切り出しのテスト。

    ガイドは約3万字あり毎回全文を渡すとコンテキストを圧迫するため、
    read_dear_ghost は「省略=目次 / 番号=その章・節 / all=全文」の3通りで返す。
    ここでは一時ファイルのサンプル文書を使い、以下を確認する:
      - 目次に番号付き見出しと最終更新日が並び、コードブロック内の `##` は拾わない
      - 章番号（'1'）指定で配下の節を含み、次の章の手前で止まる（章末の `---` は落とす）
      - 節番号（'1.1'）指定で次の見出しの手前で止まる
      - 'all' で全文、存在しない番号では目次を添えて知らせる
      - ファイルが無くても例外を投げずエラー文を返す（ツール実行を落とさない）
    """

    def _write(self, tmp_path):
        p = tmp_path / "DEAR_GHOST.md"
        p.write_text(_SAMPLE, encoding="utf-8")
        return p

    def test_omitted_section_returns_toc(self, tmp_path):
        """section 省略時は目次（見出し一覧＋最終更新）を返し、本文は含まない。"""
        out = read_dear_ghost("", path=self._write(tmp_path))
        assert "最終更新: 2026-10-06" in out
        assert "1. 記憶" in out
        assert "  1.1 種類" in out
        assert "2.1 inscribe" in out
        assert "種類の本文" not in out
        assert "9. これは" not in out

    def test_chapter_includes_subsections(self, tmp_path):
        """章指定は配下の節をすべて含み、次の章と章末の区切り線は含まない。"""
        out = read_dear_ghost("1", path=self._write(tmp_path))
        assert out.startswith("## 1. 記憶")
        assert "種類の本文" in out and "抽出の本文" in out
        assert "## 2. 道具" not in out
        assert not out.endswith("---")

    def test_subsection_stops_at_next_heading(self, tmp_path):
        """節指定は次の見出しの手前まで。コードブロック内の疑似見出しでは止まらない。"""
        out = read_dear_ghost("1.1", path=self._write(tmp_path))
        assert out.startswith("### 1.1 種類")
        assert "## 9. これはコードブロック内" in out
        assert "抽出の本文" not in out

    def test_trailing_dot_is_tolerated(self, tmp_path):
        """'2.' のように末尾ドット付きで指定されても章として解決する。"""
        out = read_dear_ghost("2.", path=self._write(tmp_path))
        assert out.startswith("## 2. 道具")

    def test_all_returns_full_text(self, tmp_path):
        """'all'（大文字小文字を問わない）で全文を返す。"""
        assert read_dear_ghost("ALL", path=self._write(tmp_path)) == _SAMPLE

    def test_unknown_section_returns_toc_with_notice(self, tmp_path):
        """存在しない番号は見つからない旨と目次を返す。"""
        out = read_dear_ghost("7.7", path=self._write(tmp_path))
        assert "「7.7」という章・節は見つかりませんでした" in out
        assert "1. 記憶" in out

    def test_missing_file_returns_error_text(self, tmp_path):
        """ファイルが開けなくても例外にせずエラー文を返す。"""
        out = read_dear_ghost("", path=tmp_path / "nope.md")
        assert out.startswith("[read_dear_ghost error:")


class TestReadDearGhostRealDocument:
    """実物の docs/explain/DEAR_GHOST.md に対するスモークテスト。

    ガイドの見出し形式（`## N.` / `### N.M`）が崩れると目次・節指定が効かなくなるため、
    実ファイルで目次が作れること・自己紹介節（2.10）が引けることを確かめる。
    """

    def test_real_toc_and_self_section(self):
        """実ファイルから目次が作れ、read_dear_ghost 自身の節が読める。"""
        assert DEAR_GHOST_PATH.exists()
        toc = read_dear_ghost("")
        assert "最終更新" in toc
        assert "2.10" in toc
        assert "read_dear_ghost" in read_dear_ghost("2.10")


class TestReadDearGhostExposure:
    """read_dear_ghost が全文脈共通の基本ツールとして台帳に載っていることのテスト。"""

    def test_in_base_tool_specs(self):
        """BASE_TOOL_SPECS に含まれ、section は任意引数である。"""
        spec = next(s for s in BASE_TOOL_SPECS if s.name == "read_dear_ghost")
        assert spec.input_schema["required"] == []
        assert "section" in spec.input_schema["properties"]
