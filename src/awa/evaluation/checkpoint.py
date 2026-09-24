from __future__ import annotations
from awa.runtime.checkpoint import CheckpointManager


def load_components_checkpoint(path, components, device="cpu"):
    modules={"world_model":components.world_model,"actor":components.actor,"uncertainty":components.uncertainty,"target_value":components.target_value}
    if getattr(components,"calibrator",None) is not None: modules["calibrator"]=components.calibrator
    if getattr(components,"arbitrator",None) is not None: modules["arbitrator"]=components.arbitrator
    if getattr(components,"q_critic",None) is not None:
        modules["q_critic"]=components.q_critic; modules["target_q"]=components.target_q; modules["q_scale"]=components.q_scale
    return CheckpointManager.load_bundle(path,modules,optimizers={},map_location=device,strict=False)
