"""Create-only publication helpers for public experiment artifacts."""
from __future__ import annotations
import ctypes
import errno
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]

def _assert_no_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink path component refused: {current}")

def validate_project_path(path: Path, *, allow_missing_leaf: bool = True) -> Path:
    """Return an absolute in-project path after checking every existing component."""
    path = path if path.is_absolute() else ROOT / path
    path = Path(os.path.abspath(path))
    _assert_no_symlink_components(path)
    root = ROOT.resolve(strict=True)
    parent = path.parent.resolve(strict=True) if path.parent.exists() else None
    if parent is None:
        raise FileNotFoundError(f"output parent directory does not exist: {path.parent}")
    if root not in (parent, *parent.parents):
        raise ValueError("path escapes the project workspace")
    if path.exists() and allow_missing_leaf:
        return path
    return path

def ensure_project_parent(path: Path) -> Path:
    """Create missing in-project parent directories without following symlinks."""
    path = path if path.is_absolute() else ROOT / path
    path = Path(os.path.abspath(path))
    _assert_no_symlink_components(path.parent)
    root = ROOT.resolve(strict=True)
    missing = []
    cursor = path.parent
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    resolved = cursor.resolve(strict=True)
    if root not in (resolved, *resolved.parents):
        raise ValueError("output parent escapes the project workspace")
    for directory in reversed(missing):
        directory.mkdir()
        if directory.is_symlink():
            raise ValueError(f"symlink directory refused: {directory}")
    _assert_no_symlink_components(path.parent)
    if root not in (path.parent.resolve(strict=True), *path.parent.resolve(strict=True).parents):
        raise ValueError("output parent escapes the project workspace")
    return path

def preflight_new_directory(path: Path) -> Path:
    """Check a create-only output directory before reading any inputs."""
    path = ensure_project_parent(path)
    if path.is_symlink():
        raise ValueError(f"output symlink refused: {path}")
    if path.exists():
        raise FileExistsError(f"output already exists; refusing overwrite: {path}")
    return path

def _write_file_exclusive(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a staged directory without replacing any path."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic RENAME_NOREPLACE is unavailable")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace_flag = 1
    result = renameat2(
        at_fdcwd, os.fsencode(source), at_fdcwd, os.fsencode(target),
        rename_noreplace_flag,
    )
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(f"output already exists; refusing overwrite: {target}")
        raise OSError(code, os.strerror(code), str(target))

def atomic_publish_directory(target: Path, files: dict[str, bytes]) -> Path:
    """Publish a complete output bundle atomically as a new directory."""
    target = ensure_project_parent(target)
    _assert_no_symlink_components(target)
    if target.exists():
        raise FileExistsError(f"output already exists; refusing overwrite: {target}")
    if not files or any(
        not name or Path(name).name != name or name in (".", "..")
        for name in files
    ):
        raise ValueError("output bundle must use nonempty plain filenames")
    lock = target.parent / f".{target.name}.publish.lock"
    lock_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    lock_fd = os.open(lock, lock_flags, 0o600)
    stage = None
    try:
        stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage.", dir=target.parent))
        for name, data in sorted(files.items()):
            if not isinstance(data, bytes):
                raise TypeError("output bundle values must be bytes")
            _write_file_exclusive(stage / name, data)
        directory_fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _rename_noreplace(stage, target)
        stage = None
        parent_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        os.close(lock_fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
    return target
