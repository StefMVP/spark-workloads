"""Exclusive disk publication. GPU leases and checkpoints are not changed."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

from clustercontrol import Refused


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def publish_missing(source, target, expected_sha256, *, usage):
    """Publish a verified missing file with an atomic no-overwrite operation."""
    source, target = Path(source), Path(target)
    if (not source.is_absolute() or not target.is_absolute() or source.is_symlink()
            or target.is_symlink() or not source.is_file() or len(expected_sha256) != 64):
        raise Refused("invalid artifact source, target or content identity")
    if any("checkpoint" in part.lower() for part in target.parts):
        raise Refused("checkpoint paths are protected")
    if any(parent.is_symlink() for parent in target.parents):
        raise Refused("artifact target has a noncanonical parent")
    if target.exists():
        if not target.is_file() or file_digest(target) != expected_sha256:
            raise Refused("existing artifact differs; overwrite is forbidden")
        return {"state": "verified", "path": str(target), "sha256": expected_sha256, "publication": "preexisting"}
    if usage not in ("new_path", "unused"):
        raise Refused("artifact usage is unknown; publication blocked")
    if file_digest(source) != expected_sha256:
        raise Refused("artifact source content differs from manifest")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".profile-prepare-", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if file_digest(temporary) != expected_sha256:
            raise Refused("staged artifact verification failed")
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or file_digest(target) != expected_sha256:
                raise Refused("artifact changed during publication")
        parent = os.open(target.parent, os.O_DIRECTORY | os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        return {"state": "verified", "path": str(target), "sha256": expected_sha256, "publication": "exclusive"}
    finally:
        temporary.unlink(missing_ok=True)
