from __future__ import annotations
from dataclasses import dataclass
import hashlib, json
import numpy as np
from .task_spec import TaskSpec

STAGE_CONCEPTS = {
    1: ('movement',),
    2: ('movement','obstacles'),
    3: ('objects','collection'),
    4: ('memory','object_permanence'),
    5: ('moving_entities','avoidance'),
    6: ('combat','damage'),
    7: ('resources','cooldowns'),
    8: ('tactics','cover','retreat'),
    9: ('multi_step','keys','doors'),
    10: ('dynamics_shift','adaptation'),
    11: ('procedural_worlds','generalization'),
    12: ('composition','transfer'),
}

STAGE_OBJECTIVES = {
    1: ('reach_goal',),
    2: ('reach_goal',),
    3: ('collect_object','reach_goal'),
    4: ('remember_goal','reach_goal'),
    5: ('avoid_enemy','reach_goal'),
    6: ('defeat_enemy','reach_goal'),
    7: ('collect_resource','defeat_enemy','reach_goal'),
    8: ('take_cover','reach_goal'),
    9: ('collect_key','open_door','reach_goal'),
    10: ('reach_goal',),
    11: ('reach_goal',),
    12: ('collect_object','collect_key','open_door','defeat_enemy','reach_goal'),
}

@dataclass
class ProceduralTaskFactory:
    base_seed: int = 0

    def make(self, stage:int, difficulty:float, index:int=0, novelty_class:str='train') -> TaskSpec:
        stage=int(stage); difficulty=float(np.clip(difficulty,0,1))
        if stage not in STAGE_CONCEPTS: raise ValueError(f'unsupported stage {stage}')
        seed=int(self.base_seed + stage*100_003 + index*997)
        rng=np.random.default_rng(seed)
        env={
            'layout_seed': int(rng.integers(0,2**31-1)),
            'obstacle_density': float(.10 + .50*difficulty if stage >= 2 else 0.0),
            'object_density': float(.15 + .55*difficulty),
            'enemy_density': float((max(stage-4,0)/8)*difficulty),
            'gravity_scale': float(rng.uniform(.65,1.35) if stage>=10 else 1.0),
            'friction_scale': float(rng.uniform(.45,1.55) if stage>=10 else 1.0),
            'movement_scale': float(rng.uniform(.75,1.25) if stage>=10 else 1.0),
            'lighting_seed': int(rng.integers(0,2**31-1)),
            'texture_seed': int(rng.integers(0,2**31-1)),
            'enemy_aggression': float(rng.uniform(.2,.95)*difficulty if stage>=5 else 0.0),
            'hide_goal_after_steps': 5 if stage == 4 else None,
        }
        goal={
            'objectives': list(STAGE_OBJECTIVES[stage]),
            'target_count':1+int(difficulty>.7),
            'success_threshold':.9,
        }
        raw=json.dumps([stage,difficulty,index,novelty_class,env,goal],sort_keys=True).encode()
        task_id=f's{stage}-{hashlib.sha256(raw).hexdigest()[:12]}'
        return TaskSpec(task_id,stage,difficulty,STAGE_CONCEPTS[stage],seed,env,goal,'engine_verifier',novelty_class)

    def batch(self, stage:int, difficulty:float, count:int, novelty_class:str='train'):
        return [self.make(stage,difficulty,i,novelty_class) for i in range(int(count))]
