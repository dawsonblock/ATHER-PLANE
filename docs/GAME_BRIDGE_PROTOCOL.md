# Aether v2.14 Game Bridge Protocol

Aether's real-game boundary is a newline-delimited JSON request/response protocol named `aether.game.v1`.
The learner does not trust the remote game process for policy or safety decisions; the bridge only transports
observations, goals, actions, rewards, termination flags and optional snapshots.

## Required operations

- `hello`: protocol negotiation and capability declaration.
- `reset`: deterministic reset when a seed is supplied.
- `step`: apply one action and return observation, goal, reward, terminated/truncated and info.
- `close`: cleanly close the session.

For counterfactual learning, implement:

- `snapshot`: return opaque simulator state bytes (base64 on the wire).
- `restore`: restore exactly those bytes and return the restored observation/goal.

## Capabilities

```json
{
  "observation_dim": 32,
  "goal_dim": 13,
  "action_dim": 4,
  "supports_snapshot": true,
  "supports_pixels": false,
  "frame_shape": null
}
```

All float arrays must be finite. Aether rejects oversized messages, protocol mismatches and non-finite rewards.
Pixel benchmarks should advertise `supports_pixels=true` and an HWC `frame_shape`, but pixel transport is expected
to use a separate shared-memory/video path in high-throughput deployments rather than embedding large frames in JSON.
