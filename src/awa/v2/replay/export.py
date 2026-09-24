from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import numpy as np


class LatentReplayExporter:
    """Export analyzed latent transitions into Aether's offline NPZ contract.

    v2.8 fixes two real-world issues:
    * hindsight-relabeled rows are excluded by default because they are not guaranteed
      to form a physically contiguous world-model trajectory;
    * when episode metadata exists, interleaved asynchronous experience is regrouped
      into episode order before export.
    """

    @staticmethod
    def _ordered_rows(rows):
        if not rows:
            return rows
        if not all((r.metadata or {}).get("episode_id") is not None for r in rows):
            return rows
        groups = OrderedDict()
        for insertion_index, r in enumerate(rows):
            md = r.metadata or {}
            groups.setdefault(str(md["episode_id"]), []).append((insertion_index, r))
        ordered = []
        for _, group in groups.items():
            if all((r.metadata or {}).get("step") is not None for _, r in group):
                group = sorted(group, key=lambda pair: (int((pair[1].metadata or {})["step"]), pair[0]))
            ordered.extend(r for _, r in group)
        return ordered

    @staticmethod
    def export(
        records,
        path: str | Path,
        *,
        include_hindsight: bool = False,
        validate_shapes: bool = True,
        order_by_episode: bool = True,
    ) -> Path:
        rows = [
            r
            for r in list(records)
            if include_hindsight or not bool((r.metadata or {}).get("hindsight", False))
        ]
        if order_by_episode:
            rows = LatentReplayExporter._ordered_rows(rows)
        if not rows:
            raise ValueError("cannot export an empty replay")

        obs_rows = [np.asarray(r.state, dtype=np.float32).reshape(-1) for r in rows]
        action_rows = [np.asarray(r.action, dtype=np.float32).reshape(-1) for r in rows]
        next_rows = [np.asarray(r.actual_next, dtype=np.float32).reshape(-1) for r in rows]
        if validate_shapes:
            obs_dim = obs_rows[0].shape
            action_dim = action_rows[0].shape
            next_dim = next_rows[0].shape
            if next_dim != obs_dim:
                raise ValueError("state and actual_next dimensions must match for world-model export")
            if any(x.shape != obs_dim for x in obs_rows) or any(x.shape != next_dim for x in next_rows):
                raise ValueError("all state/next-state rows must share one dimension")
            if any(x.shape != action_dim for x in action_rows):
                raise ValueError("all action rows must share one dimension")

        observations = np.stack(obs_rows)
        actions = np.stack(action_rows)
        rewards = np.asarray([float(r.reward) for r in rows], dtype=np.float32)
        next_observations = np.stack(next_rows)
        dones_list = []
        episode_ids = []
        for i, r in enumerate(rows):
            md = r.metadata or {}
            done = bool(md.get("done", md.get("episode_end", False)))
            episode_id = md.get("episode_id")
            episode_ids.append("" if episode_id is None else str(episode_id))
            if episode_id is not None and i + 1 < len(rows):
                next_id = (rows[i + 1].metadata or {}).get("episode_id")
                if next_id is not None and str(next_id) != str(episode_id):
                    done = True
            if episode_id is not None and i == len(rows) - 1:
                done = True
            dones_list.append(done)
        dones = np.asarray(dones_list, dtype=np.bool_)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
        )
        if any(episode_ids):
            payload["episode_ids"] = np.asarray(episode_ids)
        np.savez_compressed(path, **payload)
        return path
