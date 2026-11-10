"""Portable upload contract. Parses archives as data; never executes their contents."""
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

from .schema import ValidationError

MAX_ZIP = 128 * 1024 * 1024
MAX_EXPANDED = 512 * 1024 * 1024
MAX_FILES = 5000
MAX_CSV = 16 * 1024 * 1024


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or str(path) != name
            or any(p in (".", "..") or ":" in p or p.endswith((".", " "))
                   or re.fullmatch(r"(?i)(con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\..*)?", p)
                   for p in path.parts) or any(ord(c) < 32 for c in name)):
        raise ValidationError("Code ZIP contains an unsafe path.")
    return path


def inspect_archive(content):
    if not content or len(content) > MAX_ZIP:
        raise ValidationError("Code ZIP must be nonempty and at most 128 MiB.")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_FILES or sum(i.file_size for i in infos) > MAX_EXPANDED:
                raise ValidationError("Code ZIP exceeds 5,000 entries or 512 MiB unpacked.")
            names, files = set(), {}
            for info in infos:
                if info.orig_filename != info.filename:
                    raise ValidationError("Code ZIP contains a normalized or truncated path.")
                name = info.filename.rstrip("/") if info.is_dir() else info.filename
                safe_name(name)
                key = name.casefold()
                kind = stat.S_IFMT(info.external_attr >> 16)
                if key in names or info.flag_bits & 1 or kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValidationError("Code ZIP has duplicate paths, encrypted files, links, or special files.")
                names.add(key)
                if not info.is_dir():
                    files[name] = info
            folded_files = {name.casefold() for name in files}
            if any(str(p).casefold() in folded_files for n in names for p in PurePosixPath(n).parents if str(p) != "."):
                raise ValidationError("Code ZIP uses a file as a directory.")
            for required in ("README.md", "reproduce.json"):
                if required not in files or files[required].file_size > 65536:
                    raise ValidationError(f"Code ZIP needs {required} at its root (at most 64 KiB).")
            if not archive.read("README.md").decode("utf-8").strip():
                raise ValidationError("README.md must explain how to reproduce the entry.")
            config = json.loads(archive.read("reproduce.json"))
            if not isinstance(config, dict) or set(config) != {"image", "command"}:
                raise ValidationError("reproduce.json needs exactly image and command.")
            if not isinstance(config["image"], str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", config["image"]):
                raise ValidationError("image must name an organizer-approved environment.")
            command = config["command"]
            if (not isinstance(command, list) or not 1 <= len(command) <= 64
                    or any(not isinstance(x, str) or not x or len(x) > 4096 or "\x00" in x for x in command)):
                raise ValidationError("command must be a nonempty JSON array of at most 64 arguments.")
            # Stream every file to verify CRC/actual size without retaining an expanded archive.
            for name, info in files.items():
                count = 0
                with archive.open(info) as stream:
                    while chunk := stream.read(65536):
                        count += len(chunk)
                        if count > info.file_size:
                            raise ValidationError("ZIP file expanded beyond its declared size.")
                if count != info.file_size:
                    raise ValidationError("Truncated ZIP member.")
            return config
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError, UnicodeError, ValueError, EOFError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(f"Invalid code ZIP: {exc}") from exc


def extract_archive(content, destination):
    config = inspect_archive(content)
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            target = root / info.filename
            if not target.resolve().is_relative_to(root):
                raise ValidationError("ZIP destination escaped its directory.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("xb") as output:
                while chunk := source.read(65536):
                    output.write(chunk)
            target.chmod(0o644)
    return config


def package_code(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir() or output.exists() or output.is_relative_to(source):
        raise ValidationError("Use a source folder and a new ZIP path outside it.")
    files = []
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValidationError("Code folders cannot contain symlinks.")
        if path.is_file():
            files.append(path)
    if len(files) > MAX_FILES or sum(p.stat().st_size for p in files) > MAX_EXPANDED:
        raise ValidationError("Code folder is too large.")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            name = path.relative_to(source).as_posix()
            safe_name(name)
            archive.writestr(name, path.read_bytes())
    content = buffer.getvalue()
    inspect_archive(content)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(content)
    return output
