from __future__ import annotations
from awa.environments.delayed_cue import DelayedCueEnv
from awa.environments.robust_cue import RobustDelayedCueEnv
from awa.environments.continuous_point import ContinuousPointEnv
from awa.environments.adapters import GymnasiumDiscreteAdapter, GymnasiumContinuousAdapter, DMControlContinuousAdapter
from awa.environments.metaworld_adapter import MetaWorldContinuousAdapter


def _check_dims(env, ec):
    if env.obs_dim != ec["obs_dim"] or env.action_dim != ec["action_dim"]:
        raise ValueError(f"configured dims {(ec['obs_dim'], ec['action_dim'])} != environment {(env.obs_dim, env.action_dim)}")
    return env


def make_environment(cfg: dict, seed: int | None = None):
    ec = cfg["env"] if "env" in cfg else cfg
    kind = ec.get("kind", "delayed_cue")
    seed = int(ec.get("seed", 0) if seed is None else seed)
    if kind == "delayed_cue":
        return DelayedCueEnv(cue_delay=ec.get("cue_delay", 8), horizon=ec["horizon"], obs_dim=ec["obs_dim"], seed=seed)
    if kind == "robust_delayed_cue":
        return RobustDelayedCueEnv(cue_delay=ec.get("cue_delay", 8), horizon=ec["horizon"], obs_dim=ec["obs_dim"], seed=seed,
                                   observation_noise=float(ec.get("observation_noise", 0.0)), dropout_prob=float(ec.get("dropout_prob", 0.0)),
                                   distractor_prob=float(ec.get("distractor_prob", 0.0)))
    if kind == "continuous_point":
        return _check_dims(ContinuousPointEnv(horizon=ec["horizon"], seed=seed, step_scale=float(ec.get("step_scale", 0.15)),
                                              success_radius=float(ec.get("success_radius", 0.10)), action_dim=ec["action_dim"]), ec)
    if kind == "gym_discrete":
        return _check_dims(GymnasiumDiscreteAdapter(ec["env_id"], seed=seed, **dict(ec.get("make_kwargs", {}))), ec)
    if kind == "gym_continuous":
        return _check_dims(GymnasiumContinuousAdapter(ec["env_id"], seed=seed, **dict(ec.get("make_kwargs", {}))), ec)
    if kind == "dm_control":
        return _check_dims(DMControlContinuousAdapter(ec["domain"], ec["task"], seed=seed, task_kwargs=ec.get("task_kwargs")), ec)
    if kind == "metaworld":
        return _check_dims(MetaWorldContinuousAdapter(ec.get("task_name","reach-v3"), seed=seed), ec)
    raise ValueError(f"Unsupported env.kind: {kind}")
