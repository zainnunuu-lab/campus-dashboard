"""Dataset metadata stored as JSON "sidecar" files next to each dataset.

``campus_landuse.geojson`` -> ``campus_landuse.geojson.meta.json``

File-based sidecars keep provenance attached to the data itself, survive copying
the folder between team members, and avoid a database: the dashboard only needs
to list a few dozen datasets and read their descriptions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

from . import config

SIDECAR_SUFFIX = ".meta.json"


@dataclass
class DatasetMetadata:
    title: str
    kind: str  # "vector" | "raster" | "3d"
    category: str = "other"
    is_demo: bool = False
    description: str = ""
    source: str = ""  # e.g. "DJI Mavic 3E flight, processed in WebODM"
    capture_date: str = ""  # ISO date of the flight, if known
    processing_software: str = ""
    sensor: str = ""
    gsd_cm: float | None = None  # ground sampling distance reported by the processing tool
    crs: str = ""
    original_filename: str = ""
    accuracy_statement: str = config.DEFAULT_ACCURACY_STATEMENT
    license: str = ""
    notes: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def status_label(self) -> str:
        return config.DEMO_LABEL if self.is_demo else "Project data"

    @property
    def display_title(self) -> str:
        return f"[{config.DEMO_LABEL}] {self.title}" if self.is_demo else self.title

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetMetadata":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in known}
        clean.setdefault("title", "Untitled dataset")
        clean.setdefault("kind", "vector")
        return cls(**clean)


def sidecar_path(dataset_path: Path) -> Path:
    return dataset_path.with_name(dataset_path.name + SIDECAR_SUFFIX)


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def load_metadata(dataset_path: Path, kind: str) -> DatasetMetadata:
    """Read the sidecar, or build a minimal record when it is missing or corrupt.

    Anything stored under ``data/demo`` is always flagged as demo, whatever the
    sidecar says, so synthetic data can never be shown as a survey result.
    """
    dataset_path = Path(dataset_path)
    meta_file = sidecar_path(dataset_path)
    meta: DatasetMetadata | None = None
    if meta_file.exists():
        try:
            meta = DatasetMetadata.from_dict(json.loads(meta_file.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError, ValueError):
            meta = None
    if meta is None:
        meta = DatasetMetadata(
            title=dataset_path.stem.replace("_", " ").title(),
            kind=kind,
            notes="No metadata sidecar found - provenance unknown.",
        )
    meta.kind = kind
    if is_inside(dataset_path, config.DEMO_DIR):
        meta.is_demo = True
    if meta.category not in config.CATEGORIES:
        meta.category = "other"
    return meta


def save_metadata(dataset_path: Path, meta: DatasetMetadata) -> Path:
    target = sidecar_path(Path(dataset_path))
    target.write_text(json.dumps(meta.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return target
