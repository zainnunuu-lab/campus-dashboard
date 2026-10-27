"""Discovery of datasets on disk (demo + user folders)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import config
from .metadata import SIDECAR_SUFFIX, DatasetMetadata, is_inside, load_metadata, sidecar_path

_EXTS = {
    "vector": config.VECTOR_EXTENSIONS,
    "raster": config.RASTER_EXTENSIONS,
    "3d": config.ASSET_3D_EXTENSIONS,
}


@dataclass
class Dataset:
    path: Path
    kind: str
    meta: DatasetMetadata

    @property
    def id(self) -> str:
        return self.path.relative_to(config.DATA_DIR).as_posix()

    @property
    def is_demo(self) -> bool:
        return self.meta.is_demo

    @property
    def size_mb(self) -> float:
        return self.path.stat().st_size / 1e6

    @property
    def mtime(self) -> float:
        return self.path.stat().st_mtime


def _scan(root: Path, kind: str) -> list[Dataset]:
    folder = root / config.KIND_DIRS[kind]
    if not folder.is_dir():
        return []
    found = []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.name.endswith(SIDECAR_SUFFIX) or p.name.startswith("."):
            continue
        if p.suffix.lower() in _EXTS[kind]:
            found.append(Dataset(path=p, kind=kind, meta=load_metadata(p, kind)))
    return found


def discover(include_demo: bool = True, data_dir: Path | None = None) -> list[Dataset]:
    roots = [(data_dir or config.DATA_DIR) / "user"]
    if include_demo:
        roots.append((data_dir or config.DATA_DIR) / "demo")
    out: list[Dataset] = []
    for root in roots:
        for kind in _EXTS:
            out.extend(_scan(root, kind))
    return out


def by_category(datasets: list[Dataset], *categories: str) -> list[Dataset]:
    return [d for d in datasets if d.meta.category in categories]


def safe_filename(name: str) -> str:
    """Strip path components and unsafe characters from an uploaded filename."""
    name = Path(name).name
    stem, ext = Path(name).stem, Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "dataset"
    return f"{stem[:80]}{ext}"


def unique_path(folder: Path, filename: str) -> Path:
    target = folder / filename
    i = 1
    while target.exists():
        target = folder / f"{Path(filename).stem}_{i}{Path(filename).suffix}"
        i += 1
    return target


def delete_user_dataset(dataset: Dataset) -> None:
    """Remove a user dataset and its sidecar. Demo files are protected."""
    if dataset.is_demo or not is_inside(dataset.path, config.USER_DIR):
        raise PermissionError("Demo datasets cannot be deleted from the dashboard.")
    side = sidecar_path(dataset.path)
    dataset.path.unlink(missing_ok=True)
    side.unlink(missing_ok=True)
