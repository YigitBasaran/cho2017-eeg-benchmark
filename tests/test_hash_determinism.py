from cho2017_benchmark.data.preprocessing import PreprocessingConfig
from cho2017_benchmark.reproducibility import (
    candidate_config_hash,
    preprocessing_hash,
    split_manifest_hash,
    stable_hash,
)


def test_stable_hash_order_independent():
    assert stable_hash({"a": 1, "b": 2.0}) == stable_hash({"b": 2.0, "a": 1})
    assert stable_hash([1, 2, 3]) == stable_hash([1, 2, 3])


def test_stable_hash_is_sensitive():
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
    assert stable_hash({"a": 1}) != stable_hash({"a": 1, "b": 1})


def test_preprocessing_hash_stable_and_sensitive():
    h1 = preprocessing_hash(PreprocessingConfig().hashable())
    h2 = preprocessing_hash(PreprocessingConfig().hashable())
    assert h1 == h2
    assert h1 != preprocessing_hash(PreprocessingConfig(l_freq=7.0).hashable())


def test_split_manifest_hash_order_independent(tiny_split_manifest):
    h1 = split_manifest_hash(tiny_split_manifest)
    shuffled = tiny_split_manifest.sample(frac=1.0, random_state=3).reset_index(drop=True)
    h2 = split_manifest_hash(shuffled)
    assert h1 == h2


def test_candidate_config_hash_order_independent():
    assert candidate_config_hash({"id": "c0", "lr": 1e-3}) == candidate_config_hash({"lr": 1e-3, "id": "c0"})
