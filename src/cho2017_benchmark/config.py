"""YAML-backed experiment configuration.

Configurations are layered: ``base.yaml`` defines shared settings and each model
file (``csp_lda.yaml``, ``eegnet.yaml``, ``atcnet.yaml``) overrides the parts it
needs. :func:`resolve_config` merges the layers into an immutable
:class:`ExperimentConfig` that exposes typed convenience accessors while keeping
the full merged mapping available via ``cfg.raw`` and ``cfg.get('a.b.c')``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import paths
from .reproducibility import stable_hash


# --- YAML loading / merging ------------------------------------------------


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must define a mapping at the top level.")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* (override wins). Pure (no mutation)."""
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


_MODEL_CONFIG_FILES = {
    "csp_lda": "csp_lda.yaml",
    "eegnet": "eegnet.yaml",
    "atcnet": "atcnet.yaml",
}


def load_config(
    model_name: str | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
    config_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Load and merge ``base.yaml`` + optional model file + optional overrides."""
    cfg_dir = Path(config_dir) if config_dir is not None else paths.configs_dir()
    merged = load_yaml(cfg_dir / "base.yaml")
    if model_name is not None:
        if model_name not in _MODEL_CONFIG_FILES:
            raise KeyError(
                f"Unknown model '{model_name}'. Known: {sorted(_MODEL_CONFIG_FILES)}"
            )
        merged = deep_merge(merged, load_yaml(cfg_dir / _MODEL_CONFIG_FILES[model_name]))
    if overrides:
        merged = deep_merge(merged, overrides)
    return merged


# --- Immutable config object ----------------------------------------------


def _get_path(mapping: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = mapping
    for part in dotted.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        else:
            return default
    return node


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable view over a merged configuration mapping."""

    raw: Mapping[str, Any]

    # -- generic access --
    def get(self, dotted: str, default: Any = None) -> Any:
        return _get_path(self.raw, dotted, default)

    # -- reproducibility --
    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.raw.get("seeds", [self.seed]))

    @property
    def tuning_seed(self) -> int:
        return int(self.raw.get("tuning_seed", self.seed))

    @property
    def primary_seed(self) -> int:
        return int(self.raw.get("primary_seed", self.seed))

    @property
    def deterministic(self) -> bool:
        return bool(self.raw.get("deterministic", True))

    # -- subjects --
    @property
    def quick_mode(self) -> bool:
        return bool(self.raw.get("quick_mode", False))

    @property
    def all_subjects(self) -> tuple[int, ...]:
        subjects = self.raw.get("subjects")
        if not subjects:
            return tuple(range(1, 53))
        return tuple(int(s) for s in subjects)

    @property
    def excluded_subjects(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.raw.get("excluded_subjects", []))

    @property
    def quick_subjects(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.raw.get("quick_subjects", [1, 2, 3, 4, 5]))

    def active_subjects(self, *, include_excluded: bool = False) -> tuple[int, ...]:
        """Subjects used for modelling.

        Quick mode restricts to ``quick_subjects``. Excluded subjects (s29/s34)
        are removed unless *include_excluded* is True (used by EDA, which still
        inspects them).
        """
        base = self.quick_subjects if self.quick_mode else self.all_subjects
        if include_excluded:
            return tuple(base)
        excluded = set(self.excluded_subjects)
        return tuple(s for s in base if s not in excluded)

    def is_excluded(self, subject_index: int) -> bool:
        return int(subject_index) in set(self.excluded_subjects)

    # -- classes --
    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(self.get("classes.names", ["left_hand", "right_hand"]))

    @property
    def class_mapping(self) -> dict[str, int]:
        return dict(self.get("classes.mapping", {"left_hand": 0, "right_hand": 1}))

    # -- signal --
    @property
    def srate_in(self) -> int:
        return int(self.get("sampling.srate_in", 512))

    @property
    def srate_out(self) -> int:
        return int(self.get("sampling.srate_out", 256))

    @property
    def n_eeg(self) -> int:
        return int(self.get("channels.n_eeg", 64))

    @property
    def n_emg(self) -> int:
        return int(self.get("channels.n_emg", 4))

    @property
    def device(self) -> str:
        return str(self.raw.get("device", "auto"))

    # -- derived bundles --
    def preprocessing_dict(self) -> dict[str, Any]:
        """Canonical, hashable preprocessing parameters (see PreprocessingConfig)."""
        return {
            "reference": str(self.get("reference", "car")),
            "l_freq": float(self.get("filter.l_freq", 8.0)),
            "h_freq": float(self.get("filter.h_freq", 30.0)),
            "filter_method": str(self.get("filter.method", "iir_butter_filtfilt")),
            "iir_order": int(self.get("filter.iir_order", 4)),
            "pad_type": str(self.get("filter.pad_type", "reflect")),
            "pad_s": float(self.get("window.pad_s", 1.0)),
            "crop_start_s": float(self.get("window.crop_start_s", 0.5)),
            "crop_end_s": float(self.get("window.crop_end_s", 2.5)),
            "srate_in": self.srate_in,
            "srate_out": self.srate_out,
            "eps": float(self.get("filter.eps", 1e-7)),
        }

    def expected_n_times(self) -> int:
        """Number of samples in the final cropped + resampled task window."""
        duration = float(self.get("window.crop_end_s", 2.5)) - float(
            self.get("window.crop_start_s", 0.5)
        )
        return int(round(duration * self.srate_out))


def resolve_config(
    model_name: str | None = None,
    *,
    quick_mode: bool | None = None,
    overrides: Mapping[str, Any] | None = None,
    config_dir: Path | str | None = None,
) -> ExperimentConfig:
    """Resolve the merged configuration into an :class:`ExperimentConfig`."""
    merged = load_config(model_name, overrides=overrides, config_dir=config_dir)
    if quick_mode is not None:
        merged = deep_merge(merged, {"quick_mode": bool(quick_mode)})
    return ExperimentConfig(raw=merged)


def dump_resolved_config(cfg: ExperimentConfig, dest: Path | str) -> Path:
    """Write the fully merged configuration to *dest* as YAML."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(cfg.raw), fh, sort_keys=False, allow_unicode=True)
    return dest


def config_hash(section: Any, length: int = 16) -> str:
    """Deterministic hash of any config section/mapping."""
    return stable_hash(section, length=length)
