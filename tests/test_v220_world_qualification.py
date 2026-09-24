from pathlib import Path
import json

import torch

from awa.v2.datasets import (
    OfflineTransitionDataset,
    precompute_representation_cache,
    write_synthetic_sequence_dataset,
)
from awa.v2.representation import FrozenBackboneProjector, RepresentationCache
from awa.v2.world import MultimodalWorldModel
from awa.v2.world_training import (
    CachedFeatureSequenceDataset,
    OfflineWorldModelTrainer,
    calibrate_horizon_uncertainty,
    contiguous_sequence_starts,
    evaluate_world_model_horizons,
    validate_representation_index,
)
from awa.v2.risk_training import CachedRiskDataset, RiskConstraintTrainer, evaluate_risk_model
from awa.v2.promotion import assess_world_model_promotion


class ToyBackbone(torch.nn.Module):
    def __init__(self, in_dim=8, out_dim=12):
        super().__init__()
        torch.manual_seed(220)
        self.linear = torch.nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.linear(x.float())


def stack(tmp_path: Path, sequence_length=5):
    p = write_synthetic_sequence_dataset(
        tmp_path / "seq.npz", episodes=4, episode_length=8, obs_dim=8, action_dim=2, seed=220
    )
    ds = OfflineTransitionDataset(p)
    backbone = ToyBackbone()
    enc = FrozenBackboneProjector(backbone, 12, 6)
    cache = RepresentationCache(tmp_path / "cache", namespace=ds.manifest.sha256[:16])
    idx = precompute_representation_cache(ds, enc, cache, tmp_path / "index.json")
    seq = CachedFeatureSequenceDataset(ds, cache, idx, sequence_length)
    model = MultimodalWorldModel(6, 2, hidden=24, components=3, horizons=(1, 2, 4))
    return ds, enc, cache, idx, seq, model


def test_synthetic_sequence_dataset_is_chain_consistent_and_has_constraints(tmp_path):
    p = write_synthetic_sequence_dataset(tmp_path / "d.npz", episodes=2, episode_length=6, obs_dim=8)
    ds = OfflineTransitionDataset(p)
    assert ds.manifest.constraints_shape == (4,)
    starts = contiguous_sequence_starts(ds, 4, verify_chain=True)
    assert starts
    assert all(not ds.arrays["dones"][s : s + 3].any() for s in starts)


def test_representation_cache_can_load_by_validated_key(tmp_path):
    _, _, cache, idx, _, _ = stack(tmp_path)
    key = idx["observation_keys"][0]
    assert cache.get_by_key(key) is not None


def test_index_hash_mismatch_is_rejected(tmp_path):
    ds, _, _, idx, _, _ = stack(tmp_path)
    bad = dict(idx)
    bad["dataset_sha256"] = "0" * 64
    try:
        validate_representation_index(ds, bad)
    except ValueError as exc:
        assert "dataset hash" in str(exc)
    else:
        raise AssertionError("mismatched representation index was accepted")


def test_cached_sequence_shapes(tmp_path):
    _, _, _, _, seq, _ = stack(tmp_path)
    row = seq[0]
    assert row["features"].shape == (5, 12)
    assert row["next_features"].shape == (5, 12)
    assert row["actions"].shape == (5, 2)
    assert row["indices"].shape == (5,)


def test_world_training_and_horizon_report_are_finite(tmp_path):
    _, enc, _, _, seq, model = stack(tmp_path)
    trainer = OfflineWorldModelTrainer(model, enc.projection, lr=2e-3, overshoot_horizon=4)
    history = trainer.fit(seq, epochs=1, batch_size=8)
    assert history and all(torch.isfinite(torch.tensor(m.loss)) for m in history)
    report = evaluate_world_model_horizons(model, enc.projection, seq, (1, 2, 4), batch_size=8)
    assert report.samples > 0
    assert set(report.horizon_rmse) == {"1", "2", "4"}
    assert all(torch.isfinite(torch.tensor(v)) for v in report.horizon_rmse.values())


def test_uncertainty_calibration_is_horizon_shaped(tmp_path):
    _, enc, _, _, seq, model = stack(tmp_path)
    trainer = OfflineWorldModelTrainer(model, enc.projection, lr=1e-3, overshoot_horizon=4)
    trainer.fit(seq, epochs=1, batch_size=8)
    result = calibrate_horizon_uncertainty(model, enc.projection, seq, epochs=1, batch_size=8)
    assert set(result["mae_by_horizon"]) == {"1", "2", "4"}
    assert result["mean_mae"] >= 0


def test_risk_training_uses_explicit_constraint_labels(tmp_path):
    _, enc, _, _, seq, model = stack(tmp_path)
    risk_ds = CachedRiskDataset(seq)
    before = evaluate_risk_model(model.risk, enc.projection, risk_ds, batch_size=8)
    trainer = RiskConstraintTrainer(model.risk, enc.projection, lr=2e-3)
    history = trainer.fit(risk_ds, epochs=1, batch_size=8)
    after = evaluate_risk_model(model.risk, enc.projection, risk_ds, batch_size=8)
    assert history
    assert 0 <= before.brier <= 1
    assert 0 <= after.accuracy <= 1


def test_promotion_gate_rejects_large_horizon_regression():
    baseline = {"one_step_nll": 2.0, "horizon_rmse": {"1": 1.0, "5": 2.0}}
    candidate = {"one_step_nll": 1.9, "horizon_rmse": {"1": 0.9, "5": 2.4}}
    decision = assess_world_model_promotion(candidate, baseline, max_horizon_regression_pct=3.0)
    assert not decision.promoted
    assert decision.worst_horizon_regression_pct > 3.0


def test_promotion_gate_accepts_nonregressing_candidate():
    baseline = {"one_step_nll": 2.0, "horizon_rmse": {"1": 1.0, "5": 2.0}}
    candidate = {"one_step_nll": 1.8, "horizon_rmse": {"1": 0.9, "5": 1.8}}
    decision = assess_world_model_promotion(candidate, baseline, min_average_horizon_improvement_pct=5.0)
    assert decision.promoted
