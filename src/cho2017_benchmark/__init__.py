"""Cho2017 Motor Imagery EEG Benchmark.

Reproducible comparison of a traditional CSP + LDA pipeline against two
deep-learning models (EEGNet, ATCNet) on the Cho et al. 2017 GigaDB
motor-imagery dataset.

The package is organized into sub-packages:

- :mod:`cho2017_benchmark.data` -- download, inspection, loading, epoching,
  artifact bookkeeping, preprocessing, processed-cache preparation and splits.
- :mod:`cho2017_benchmark.models` -- EEGNet / ATCNet construction behind a
  central factory.
- :mod:`cho2017_benchmark.training` -- generic PyTorch trainer, callbacks and
  checkpoint helpers.
- :mod:`cho2017_benchmark.evaluation` -- metrics, predictions, latency,
  statistics and plotting.
- :mod:`cho2017_benchmark.reporting` -- schema-validated result writers and the
  cross-model comparison utilities.

Top-level helpers live in :mod:`cho2017_benchmark.paths`,
:mod:`cho2017_benchmark.config` and :mod:`cho2017_benchmark.reproducibility`.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
