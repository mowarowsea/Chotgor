"""debug_log_archiver — debug/ 生ログの月次退避のユニットテスト。

背景:
    CHOTGOR_DEBUG=1 で運用していると debug/{request_id}/ が際限なく積み上がる
    （応答の再生成を重ねたシナリオ運用では1日20MB近い）。削除の仕組みが無く、
    実際に 1.1GB / 13,761 リクエスト分まで膨らんだのが導入の動機。
    月替わりで「当月・前月より古い月」を tar.gz へ畳み、元データを消す。

検証対象:
    - 当月・前月を残し、それ以前だけを畳む月境の判定（年またぎ含む）
    - 検証（tar 内エントリ数と実体の一致）が通らない月は元データを消さない安全弁
    - 対象が無ければ何もしない冪等性（日次スケジューラから毎日呼ばれるため必須）
    - 同じ月を二度畳む状況で既存アーカイブを壊さない連番
    - アーカイブから元の debug/{request_id}/ 構造へ復元できること
"""

import os
import tarfile
from datetime import datetime
from pathlib import Path

from backend.lib import debug_log_archiver
from backend.lib.debug_log_archiver import archive_debug_logs

# 判定の基準時刻。当月=2026-08 / 前月=2026-07 が保持され、6月以前が退避対象になる
NOW = datetime(2026, 8, 19, 12, 0, 0)


def _make_log_dir(root: Path, name: str, when: datetime, file_count: int = 2) -> Path:
    """debug/{request_id}/ を模したディレクトリを、指定の更新時刻で作る。

    退避対象の判定はディレクトリの mtime で行うため、中身のファイルを作った後に
    ディレクトリ側の mtime を上書きする順序が重要（ファイル作成で親の mtime が動く）。
    """
    d = root / name
    d.mkdir(parents=True)
    for i in range(file_count):
        (d / f"{i:02d}_chat_Request_Sonnet.log").write_text(f"dummy {name} {i}", encoding="utf-8")
    ts = when.timestamp()
    for f in d.iterdir():
        os.utime(f, (ts, ts))
    os.utime(d, (ts, ts))
    return d


def _standard_tree(root: Path) -> None:
    """退避対象2ヶ月（5月・6月）と保持対象2ヶ月（7月・8月）を並べた標準構成。"""
    _make_log_dir(root, "aaaa1111", datetime(2026, 5, 10))
    _make_log_dir(root, "bbbb2222", datetime(2026, 6, 15))
    _make_log_dir(root, "cccc3333", datetime(2026, 7, 20))
    _make_log_dir(root, "dddd4444", datetime(2026, 8, 5))


# ─── 月境の判定 ───────────────────────────────────────────────────────────────

def test_当月と前月を残しそれ以前だけを畳む(tmp_path):
    """5月・6月は tar.gz になって消え、7月（前月）・8月（当月）は手つかずで残る。"""
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    _standard_tree(debug_dir)

    archived = archive_debug_logs(debug_dir, now=NOW)

    assert archived == ["2026-05", "2026-06"]
    assert not (debug_dir / "aaaa1111").exists()
    assert not (debug_dir / "bbbb2222").exists()
    assert (debug_dir / "cccc3333").is_dir()
    assert (debug_dir / "dddd4444").is_dir()
    archive_dir = debug_dir / "_archive"
    assert (archive_dir / "debug_202605.tar.gz").is_file()
    assert (archive_dir / "debug_202606.tar.gz").is_file()


def test_年またぎでも前月を正しく残す(tmp_path):
    """1月時点なら前月は前年12月。単純な month-1 だと 0 月になって壊れる境界。"""
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    _make_log_dir(debug_dir, "dec00001", datetime(2025, 12, 20))
    _make_log_dir(debug_dir, "nov00001", datetime(2025, 11, 20))
    _make_log_dir(debug_dir, "jan00001", datetime(2026, 1, 5))

    archived = archive_debug_logs(debug_dir, now=datetime(2026, 1, 15))

    assert archived == ["2025-11"]
    assert (debug_dir / "dec00001").is_dir()
    assert (debug_dir / "jan00001").is_dir()
    assert not (debug_dir / "nov00001").exists()


def test_同じ月の複数ディレクトリが1つのアーカイブにまとまる(tmp_path):
    """月単位で畳むので、同月のリクエストは何件あっても tar.gz は1本。"""
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    for i in range(5):
        _make_log_dir(debug_dir, f"may0000{i}", datetime(2026, 5, 10 + i))

    archived = archive_debug_logs(debug_dir, now=NOW)

    assert archived == ["2026-05"]
    assert len(list((debug_dir / "_archive").glob("*.tar.gz"))) == 1
    with tarfile.open(debug_dir / "_archive" / "debug_202605.tar.gz") as tf:
        names = tf.getnames()
    assert sum(1 for n in names if n.endswith("_chat_Request_Sonnet.log")) == 10


# ─── 冪等性・対象なし ─────────────────────────────────────────────────────────

def test_対象が無ければ何もしない(tmp_path):
    """日次スケジューラから毎日呼ばれるため、対象なしで空振りできることが前提。"""
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    _make_log_dir(debug_dir, "cccc3333", datetime(2026, 7, 20))

    assert archive_debug_logs(debug_dir, now=NOW) == []
    assert not (debug_dir / "_archive").exists()


def test_二度目の実行は空振りする(tmp_path):
    """畳んだ直後に再実行しても、対象が無くなっているので何も起きない。"""
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    _standard_tree(debug_dir)

    assert archive_debug_logs(debug_dir, now=NOW) == ["2026-05", "2026-06"]
    assert archive_debug_logs(debug_dir, now=NOW) == []
    assert len(list((debug_dir / "_archive").glob("*.tar.gz"))) == 2


def test_debugディレクトリが無ければ空を返す(tmp_path):
    """CHOTGOR_DEBUG=0 で運用していれば debug/ 自体が存在しない。"""
    assert archive_debug_logs(tmp_path / "not_exist", now=NOW) == []


def test_アーカイブ置き場自身は退避対象にならない(tmp_path):
    """_archive の mtime が古くても、自分自身を tar に巻き込んではいけない。"""
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    archive_dir = debug_dir / "_archive"
    archive_dir.mkdir()
    (archive_dir / "debug_202603.tar.gz").write_bytes(b"existing archive")
    os.utime(archive_dir, (datetime(2026, 3, 1).timestamp(),) * 2)
    _make_log_dir(debug_dir, "aaaa1111", datetime(2026, 5, 10))

    archived = archive_debug_logs(debug_dir, now=NOW)

    assert archived == ["2026-05"]
    assert (archive_dir / "debug_202603.tar.gz").read_bytes() == b"existing archive"


# ─── 安全弁 ───────────────────────────────────────────────────────────────────

def test_検証が通らなければ元データを消さない(tmp_path, monkeypatch):
    """tar と実体のエントリ数がズレたら削除を中止する。

    圧縮が途中で壊れた状態で生ログだけ消えると復元不能になるため、
    「消していいか」の判断は必ず突き合わせ結果に従わせる。
    """
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    _make_log_dir(debug_dir, "aaaa1111", datetime(2026, 5, 10))

    monkeypatch.setattr(debug_log_archiver, "_count_entries_on_disk", lambda dirs: 999)
    archived = archive_debug_logs(debug_dir, now=NOW)

    assert archived == []
    assert (debug_dir / "aaaa1111").is_dir()
    assert (debug_dir / "aaaa1111" / "00_chat_Request_Sonnet.log").is_file()


def test_圧縮に失敗したら壊れたアーカイブを残さない(tmp_path, monkeypatch):
    """書き込み途中で落ちた tar.gz を放置すると、次回の連番判定を汚すだけで復元にも使えない。"""
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    _make_log_dir(debug_dir, "aaaa1111", datetime(2026, 5, 10))

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(tarfile.TarFile, "add", _boom)
    archived = archive_debug_logs(debug_dir, now=NOW)

    assert archived == []
    assert (debug_dir / "aaaa1111").is_dir()
    assert list((debug_dir / "_archive").glob("*.tar.gz")) == []


def test_同月を二度畳んでも既存アーカイブを上書きしない(tmp_path):
    """畳んだ後にその月の生ログが再び現れた場合、連番を振って別ファイルにする。"""
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    _make_log_dir(debug_dir, "aaaa1111", datetime(2026, 5, 10))
    archive_debug_logs(debug_dir, now=NOW)
    first = (debug_dir / "_archive" / "debug_202605.tar.gz").read_bytes()

    _make_log_dir(debug_dir, "aaaa2222", datetime(2026, 5, 11))
    archived = archive_debug_logs(debug_dir, now=NOW)

    assert archived == ["2026-05"]
    assert (debug_dir / "_archive" / "debug_202605.tar.gz").read_bytes() == first
    assert (debug_dir / "_archive" / "debug_202605_2.tar.gz").is_file()


# ─── 復元 ─────────────────────────────────────────────────────────────────────

def test_アーカイブから元の構造へ復元できる(tmp_path):
    """tar 内のパスが debug/{request_id}/... なので、展開すれば元の場所へ戻る。"""
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    _make_log_dir(debug_dir, "aaaa1111", datetime(2026, 5, 10))
    original = (debug_dir / "aaaa1111" / "00_chat_Request_Sonnet.log").read_text(encoding="utf-8")

    archive_debug_logs(debug_dir, now=NOW)
    assert not (debug_dir / "aaaa1111").exists()

    with tarfile.open(debug_dir / "_archive" / "debug_202605.tar.gz") as tf:
        tf.extractall(tmp_path)

    restored = Path(tmp_path / "debug" / "aaaa1111" / "00_chat_Request_Sonnet.log")
    assert restored.read_text(encoding="utf-8") == original
