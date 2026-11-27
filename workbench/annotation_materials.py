"""Preserve the source bytes of annotations accepted during image import."""
from contextlib import contextmanager, suppress
from hashlib import file_digest, sha256
import os
from pathlib import Path
import re
import stat
from tempfile import NamedTemporaryFile

from .project_layout import project_reference
from .store import ConflictError


def _require_owned_path(path):
    if path.is_symlink() or (path.exists()
                            and getattr(path.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        raise ValueError('標註素材儲存路徑不得為連結')


def _preserve(folder, source, expected, created):
    extension = source.suffix.lower()
    if extension not in {'.json', '.jsonl', '.txt', '.yaml', '.yml'} or not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError('匯入標註檔案格式或雜湊無效')
    target = project_reference(folder, f'imported-annotations/{expected}{extension}')
    directory = target.parent
    _require_owned_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _require_owned_path(target)
    temporary = None
    try:
        try:
            incoming = source.open('rb')
        except OSError as error:
            raise ConflictError(f'無法讀取原標註檔 {source.name}，請重新預覽') from error
        with incoming:
            if target.is_file():
                incoming_digest = file_digest(incoming, 'sha256').hexdigest()
                with target.open('rb') as saved:
                    if file_digest(saved, 'sha256').hexdigest() != expected:
                        raise ValueError(f'已存標註檔雜湊不符：{target.name}')
            else:
                digest = sha256()
                with NamedTemporaryFile(dir=directory, suffix='.tmp', delete=False) as output:
                    temporary = Path(output.name)
                    while chunk := incoming.read(1024 * 1024):
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                incoming_digest = digest.hexdigest()
        if incoming_digest != expected:
            raise ConflictError(f'來源標註 {source.name} 已變更，請重新預覽')
        if temporary is not None:
            temporary.replace(target)
            created.append(target)
        return {'relative_path': target.relative_to(folder).as_posix(),
                'sha256': expected, 'byte_count': target.stat().st_size}
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


@contextmanager
def preserve_annotation_sources(folder):
    """Cache shared input files once per batch and remove newly staged files on rollback."""
    saved, created = {}, []

    def preserve(descriptor):
        source, digest = Path(descriptor['path']), descriptor['sha256']
        key = (source, digest)
        if key not in saved:
            saved[key] = _preserve(folder, source, digest, created)
        return saved[key]

    try:
        yield preserve
    except BaseException:
        for path in created:
            with suppress(OSError):
                path.unlink(missing_ok=True)
        raise
