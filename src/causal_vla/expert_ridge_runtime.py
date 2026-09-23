"""Regularized source-free prediction of causally localized expert residuals."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import torch
from torch import Tensor

from causal_vla.action_flow import RidgeFlowRegressor
from causal_vla.causal_trace_runtime import PrefixSnapshot
from causal_vla.expert_knn_runtime import ExpertPrototypeBank, load_expert_prototype_bank
from causal_vla.expert_steering_runtime import ExpertSteeringSpec
from causal_vla.expert_trace_runtime import expert_layers


class RidgeCrossValidationRow(TypedDict):
    alpha: float
    scale: float
    loo_target_mse_recovery: float
    loo_target_cosine: float


@dataclass(frozen=True)
class ExpertRidgeBank:
    """One dual ridge map per denoising step at a frozen expert-residual locus."""

    spec: ExpertSteeringSpec
    num_steps: int
    regressors: dict[int, RidgeFlowRegressor]
    alpha: float
    scale: float

    def __post_init__(self) -> None:
        if self.num_steps <= 0 or set(self.regressors) != set(range(self.num_steps)):
            raise ValueError("Ridge regressors must cover every denoising step")
        if self.alpha <= 0 or not torch.isfinite(torch.tensor(self.scale)):
            raise ValueError("Ridge alpha and scale must be valid")


@dataclass(frozen=True)
class ExpertRidgeFitReport:
    schema_version: int
    source_prototype_bank: str
    spec: dict[str, object]
    sample_count_per_flow_step: int
    alpha_candidates: tuple[float, ...]
    selected_alpha: float
    selected_scale: float
    cross_validation: tuple[RidgeCrossValidationRow, ...]

    def to_dict(self) -> dict[str, object]:
        payload = cast(dict[str, object], asdict(self))
        payload["cross_validation"] = list(self.cross_validation)
        return payload


def _regressor_payload(regressor: RidgeFlowRegressor) -> dict[str, object]:
    return {
        "mean": regressor.mean,
        "scale": regressor.scale,
        "target_mean": regressor.target_mean,
        "normalized_keys": regressor.normalized_keys,
        "dual_weights": regressor.dual_weights,
        "alpha": regressor.alpha,
    }


def _expert_ridge_payload(bank: ExpertRidgeBank) -> dict[str, object]:
    """Return the tensor-only payload shared by standalone and composite banks."""

    return {
        "schema_version": 1,
        "controller_type": "ridge",
        "spec": asdict(bank.spec),
        "num_steps": bank.num_steps,
        "alpha": bank.alpha,
        "scale": bank.scale,
        "regressors": {
            step: _regressor_payload(regressor) for step, regressor in bank.regressors.items()
        },
    }


def save_expert_ridge_bank(bank: ExpertRidgeBank, path: str | Path) -> None:
    """Serialize a tensor-only ridge bank without arbitrary Python objects."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _expert_ridge_payload(bank)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def _expert_ridge_from_payload(payload: object) -> ExpertRidgeBank:
    """Validate and construct a ridge bank from a tensor-only payload."""

    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("controller_type") != "ridge"
    ):
        raise ValueError("Unsupported expert ridge bank schema")
    spec_payload = payload.get("spec")
    raw_regressors = payload.get("regressors")
    if not isinstance(spec_payload, dict) or not isinstance(raw_regressors, dict):
        raise ValueError("Expert ridge bank is missing required fields")
    spec = ExpertSteeringSpec(
        name=str(spec_payload["name"]),
        layer=int(spec_payload["layer"]),
        token_positions=tuple(int(value) for value in spec_payload["token_positions"]),
        interface=cast(Any, spec_payload["interface"]),
    )
    regressors: dict[int, RidgeFlowRegressor] = {}
    for raw_step, raw in raw_regressors.items():
        if not isinstance(raw_step, int) or not isinstance(raw, dict):
            raise ValueError("Expert ridge regressor payload is malformed")
        tensors = tuple(
            raw.get(name)
            for name in (
                "mean",
                "scale",
                "target_mean",
                "normalized_keys",
                "dual_weights",
            )
        )
        if not all(isinstance(value, Tensor) for value in tensors):
            raise ValueError("Expert ridge regressor tensors are malformed")
        regressors[raw_step] = RidgeFlowRegressor(
            cast(Tensor, tensors[0]),
            cast(Tensor, tensors[1]),
            cast(Tensor, tensors[2]),
            cast(Tensor, tensors[3]),
            cast(Tensor, tensors[4]),
            float(raw["alpha"]),
        )
    return ExpertRidgeBank(
        spec=spec,
        num_steps=int(payload["num_steps"]),
        regressors=regressors,
        alpha=float(payload["alpha"]),
        scale=float(payload["scale"]),
    )


def load_expert_ridge_bank(path: str | Path) -> ExpertRidgeBank:
    """Load and validate a tensor-only expert ridge bank."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return _expert_ridge_from_payload(payload)


def _ridge_loo_predictions(bank: ExpertPrototypeBank, *, alpha: float) -> tuple[Tensor, Tensor]:
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    for step in range(bank.num_steps):
        features = bank.features[step]
        directions = bank.directions[step]
        episodes = bank.episode_indices[step]
        for heldout_episode in torch.unique(episodes):
            train = episodes != heldout_episode
            test = ~train
            regressor = RidgeFlowRegressor.fit(features[train], directions[train], alpha=alpha)
            predictions.extend(
                regressor.predict(feature).reshape(1, -1) for feature in features[test]
            )
            targets.append(directions[test].reshape(int(test.sum()), -1))
    return torch.cat(predictions).double(), torch.cat(targets).double()


def fit_expert_ridge_bank(
    prototype_bank: ExpertPrototypeBank,
    *,
    alpha_candidates: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0),
) -> tuple[ExpertRidgeBank, tuple[RidgeCrossValidationRow, ...]]:
    """Select ridge regularization by leave-one-training-episode-out recovery."""

    if not alpha_candidates or len(set(alpha_candidates)) != len(alpha_candidates):
        raise ValueError("Ridge alpha candidates must be nonempty and unique")
    if any(alpha <= 0 for alpha in alpha_candidates):
        raise ValueError("Ridge alpha candidates must be positive")
    rows: list[RidgeCrossValidationRow] = []
    for alpha in alpha_candidates:
        prediction, target = _ridge_loo_predictions(prototype_bank, alpha=alpha)
        denominator = torch.sum(prediction * prediction)
        if float(denominator) <= 1e-20:
            raise ValueError("Ridge cross-validation predictions are degenerate")
        scale = float(torch.sum(prediction * target) / denominator)
        residual = scale * prediction - target
        recovery = 1.0 - float(torch.sum(residual * residual) / torch.sum(target * target))
        cosine = float(
            torch.nn.functional.cosine_similarity(
                (scale * prediction).reshape(-1), target.reshape(-1), dim=0
            )
        )
        rows.append(
            {
                "alpha": alpha,
                "scale": scale,
                "loo_target_mse_recovery": recovery,
                "loo_target_cosine": cosine,
            }
        )
    selected = max(rows, key=lambda row: row["loo_target_mse_recovery"])
    alpha = selected["alpha"]
    regressors = {
        step: RidgeFlowRegressor.fit(
            prototype_bank.features[step],
            prototype_bank.directions[step],
            alpha=alpha,
        )
        for step in range(prototype_bank.num_steps)
    }
    bank = ExpertRidgeBank(
        spec=prototype_bank.spec,
        num_steps=prototype_bank.num_steps,
        regressors=regressors,
        alpha=alpha,
        scale=selected["scale"],
    )
    return bank, tuple(rows)


class AdaptiveExpertRidge:
    """Predict and add a ridge correction from the current expert residual."""

    def __init__(
        self,
        bank: ExpertRidgeBank,
        flow_step: int,
        *,
        direction_sign: float = 1.0,
        random_seed: int | None = None,
    ):
        if direction_sign not in {-1.0, 1.0}:
            raise ValueError("Direction sign must be +1 or -1")
        self.bank = bank
        self.flow_step = flow_step
        self.direction_sign = direction_sign
        self.random_seed = random_seed
        self.residual_input: Tensor | None = None
        self.calls = 0

    def capture_residual(self, _module: Any, inputs: tuple[object, ...]) -> None:
        if not inputs or not isinstance(inputs[0], Tensor):
            raise TypeError("Expected a tensor residual input")
        if self.residual_input is not None:
            raise RuntimeError("Expert residual was captured more than once")
        self.residual_input = inputs[0].detach().clone()

    def add_prediction(self, _module: Any, _inputs: tuple[object, ...], output: object) -> Tensor:
        if not isinstance(output, Tensor):
            raise TypeError("Expected a tensor expert MLP output")
        if self.bank.spec.interface == "post_residual" and self.residual_input is None:
            raise RuntimeError("Expert MLP ran before residual capture")
        if self.calls:
            raise RuntimeError("Expert ridge hooks ran out of order")
        self.calls += 1
        positions = list(self.bank.spec.token_positions)
        activation = (
            output
            if self.bank.spec.interface == "expert_output"
            else cast(Tensor, self.residual_input) + output
        )
        selected = activation[:, positions]
        prediction = self.bank.regressors[self.flow_step].predict(selected).reshape_as(selected)
        if self.random_seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(self.random_seed)
            random_direction = torch.randn(
                prediction.shape,
                dtype=torch.float32,
                device="cpu",
                generator=generator,
            )
            prediction_norm = torch.linalg.vector_norm(prediction.float().cpu())
            prediction = random_direction * (
                prediction_norm / torch.linalg.vector_norm(random_direction).clamp_min(1e-12)
            )
        prediction = prediction * self.direction_sign
        updated = output.clone()
        updated[:, positions] = (
            output[:, positions]
            + prediction.to(device=output.device, dtype=output.dtype) * self.bank.scale
        )
        return updated

    def assert_applied(self) -> None:
        residual_missing = (
            self.bank.spec.interface == "post_residual" and self.residual_input is None
        )
        if residual_missing or self.calls != 1:
            raise RuntimeError("Adaptive expert ridge was not applied exactly once")


def sample_with_expert_ridge(
    policy: Any,
    snapshot: PrefixSnapshot,
    noise: Tensor,
    bank: ExpertRidgeBank,
    *,
    direction_sign: float = 1.0,
    random_seed: int | None = None,
) -> Tensor:
    """Integrate an action chunk with ridge-predicted expert-residual corrections."""

    model = policy.model
    layers, final_norm = expert_layers(policy)
    if bank.spec.layer >= len(layers) or bank.num_steps != int(model.config.num_steps):
        raise ValueError("Expert ridge bank is incompatible with the policy")
    if bank.spec.interface == "expert_output" and bank.spec.layer != len(layers) - 1:
        raise ValueError("Expert-output steering requires the final expert layer")
    x_t = noise.clone()
    dt = -1.0 / bank.num_steps
    for flow_step in range(bank.num_steps):
        timestep = torch.full(
            (x_t.shape[0],),
            1.0 + flow_step * dt,
            dtype=torch.float32,
            device=x_t.device,
        )
        intervention = AdaptiveExpertRidge(
            bank,
            flow_step,
            direction_sign=direction_sign,
            random_seed=(None if random_seed is None else random_seed + flow_step),
        )
        if bank.spec.interface == "expert_output":
            residual_handle = None
            prediction_handle = final_norm.register_forward_hook(intervention.add_prediction)
        else:
            layer = layers[bank.spec.layer]
            residual_handle = layer.post_attention_layernorm.register_forward_pre_hook(
                intervention.capture_residual
            )
            prediction_handle = layer.mlp.register_forward_hook(intervention.add_prediction)
        try:
            with torch.inference_mode():
                velocity = model.denoise_step(
                    prefix_pad_masks=snapshot.pad_masks,
                    past_key_values=snapshot.cache,
                    x_t=x_t,
                    timestep=timestep,
                )
        finally:
            if residual_handle is not None:
                residual_handle.remove()
            prediction_handle.remove()
        intervention.assert_applied()
        x_t = x_t + dt * velocity
    action_dim = int(policy.config.action_feature.shape[0])
    return x_t[..., :action_dim]


def _positive_floats(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("Expected comma-separated positive floats")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m causal_vla.expert_ridge_runtime")
    parser.add_argument("--prototype-bank", type=Path, required=True)
    parser.add_argument(
        "--alpha-candidates", type=_positive_floats, default=(0.001, 0.01, 0.1, 1.0)
    )
    parser.add_argument("--bank-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = load_expert_prototype_bank(args.prototype_bank)
    bank, rows = fit_expert_ridge_bank(source, alpha_candidates=args.alpha_candidates)
    save_expert_ridge_bank(bank, args.bank_output)
    report = ExpertRidgeFitReport(
        schema_version=1,
        source_prototype_bank=str(args.prototype_bank.resolve()),
        spec=asdict(bank.spec),
        sample_count_per_flow_step=int(source.features[0].shape[0]),
        alpha_candidates=args.alpha_candidates,
        selected_alpha=bank.alpha,
        selected_scale=bank.scale,
        cross_validation=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
