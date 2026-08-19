"""debug/ の生ログを月単位で tar.gz へ退避するモジュール。

CHOTGOR_DEBUG=1 で走らせているとリクエスト1件あたり数十〜数百KBの生ログが
debug/{request_id}/ に積まれ続ける（応答の再生成を重ねると1日20MB近くになる）。
削除の仕組みが無いままだと際限なく膨らむため、当月・前月を残してそれ以前の月を
月単位のアーカイブへ畳む。

安全側の設計:
    tar 内のエントリ数と実体のエントリ数が一致した月だけ元データを削除する。
    一致しなければアーカイブも元データも残すので、圧縮が壊れた状態で生ログだけ
    消える事故は起きない。

Logs UI との関係:
    debug_log_entries（SQLite）には触らない。一覧・発話・応答・reasoning は DB 側に
    残り続け、raw_dir が指す先が消えても api/logs_ui/entries.py の raw_path.exists()
    ガードで素通りする。畳まれた月は生ファイルの閲覧だけができなくなる。

復元:
    tar -xzf debug/_archive/debug_YYYYMM.tar.gz  （元の debug/{request_id}/ へ戻る）
"""

import logging
import shutil
import tarfile
from datetime import datetime, timedelta
from pathlib import Path

from backend.lib.debug_logger import ChotgorLogger

_log = logging.getLogger(__name__)

DEBUG_DIR = Path(ChotgorLogger.DEBUG_DIR)
ARCHIVE_DIRNAME = "_archive"

# gzip の圧縮レベル。9 は 6 に対して数%しか縮まないのに所要時間が跳ね上がるため 6 を使う
_COMPRESS_LEVEL = 6


def _month_key(path: Path) -> str:
    """ディレクトリの更新時刻から "YYYY-MM" を返す。"""
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m")


def _months_to_keep(now: datetime) -> set[str]:
    """退避対象から外す月（当月・前月）を返す。"""
    current = now.replace(day=1)
    previous = current - timedelta(days=1)
    return {current.strftime("%Y-%m"), previous.strftime("%Y-%m")}


def _next_free_path(archive_dir: Path, year_month: str) -> Path:
    """空いている連番のアーカイブパスを返す。

    同じ月を二度退避する状況（畳んだ後にその月の生ログが再び現れた等）でも
    既存アーカイブを上書きしないよう、debug_YYYYMM_2.tar.gz と番号を振る。
    """
    candidate = archive_dir / f"debug_{year_month}.tar.gz"
    seq = 2
    while candidate.exists():
        candidate = archive_dir / f"debug_{year_month}_{seq}.tar.gz"
        seq += 1
    return candidate


def _count_entries_on_disk(dirs: list[Path]) -> int:
    """実体のエントリ数（ディレクトリ自身＋配下の全ファイル・全サブディレクトリ）。"""
    return sum(1 + sum(1 for _ in d.rglob("*")) for d in dirs)


def _archive_month(month: str, dirs: list[Path], archive_dir: Path, root_name: str) -> bool:
    """1ヶ月分を圧縮・検証し、検証が通ったときだけ元データを削除する。

    Returns:
        元データの削除まで完了したら True。圧縮失敗・検証不一致なら False。
    """
    tarball = _next_free_path(archive_dir, month.replace("-", ""))
    try:
        with tarfile.open(tarball, "w:gz", compresslevel=_COMPRESS_LEVEL) as tf:
            for d in dirs:
                tf.add(d, arcname=f"{root_name}/{d.name}")
    except Exception:
        _log.exception("デバッグログ退避 圧縮失敗 month=%s（元データは残す）", month)
        tarball.unlink(missing_ok=True)
        return False

    with tarfile.open(tarball, "r:gz") as tf:
        entries_in_tar = len(tf.getnames())
    entries_on_disk = _count_entries_on_disk(dirs)
    if entries_in_tar != entries_on_disk:
        _log.error(
            "デバッグログ退避 検証不一致 month=%s tar=%d disk=%d（元データは残す）",
            month, entries_in_tar, entries_on_disk,
        )
        return False

    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
    _log.info(
        "デバッグログ退避 完了 month=%s dirs=%d entries=%d archive=%.1fMB",
        month, len(dirs), entries_in_tar, tarball.stat().st_size / 1048576,
    )
    return True


def archive_debug_logs(debug_dir: Path | str = DEBUG_DIR, *, now: datetime | None = None) -> list[str]:
    """当月・前月を除く月の生ログを debug/_archive/ へ畳む。

    ブロッキングな重い処理（数百MB規模で数分かかる）なので、
    イベントループ上ではなく asyncio.to_thread() 経由で呼ぶこと。

    Args:
        debug_dir: 生ログの親ディレクトリ。
        now: 当月・前月の判定に使う基準時刻（省略時は現在時刻）。

    Returns:
        元データの削除まで完了した月のリスト（"YYYY-MM"）。対象なしなら空。
    """
    debug_path = Path(debug_dir)
    if not debug_path.is_dir():
        return []

    keep = _months_to_keep(now or datetime.now())
    targets: dict[str, list[Path]] = {}
    for entry in debug_path.iterdir():
        if not entry.is_dir() or entry.name == ARCHIVE_DIRNAME:
            continue
        month = _month_key(entry)
        if month in keep:
            continue
        targets.setdefault(month, []).append(entry)

    if not targets:
        return []

    archive_dir = debug_path / ARCHIVE_DIRNAME
    archive_dir.mkdir(parents=True, exist_ok=True)
    return [
        month for month in sorted(targets)
        if _archive_month(month, sorted(targets[month]), archive_dir, debug_path.name)
    ]
