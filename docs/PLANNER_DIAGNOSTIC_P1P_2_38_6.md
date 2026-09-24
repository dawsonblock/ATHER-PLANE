# P1P planner diagnosis (v2.38.6)

P1P follows the small paired reward experiment (P1D) and blocks P2 until the
planner questions have actual answers. It does not train a larger model or
change MPPI/iCEM. The raw file is intentionally an evidence artifact: the
validator cannot make up simulator returns or a task reward oracle.

## Fixed comparison

Use one frozen `my_way_home` checkpoint pair and the same held-out episode seeds,
starting states, and action-sequence populations for all four tests.
Evaluate horizons `1, 2, 4, 8, 16, 32` and both `actor_seeded` and `mixed`
populations. The mixed population must include actor and random proposals. Record
which candidates are actor, random, or elites. Do not retune the candidate
distribution separately for each test.

| Test | Dynamics | Reward | Terminal value | Risk |
| --- | --- | --- | --- | --- |
| P1P-A | real ViZDoom branch | environment oracle | off | off |
| P1P-B | learned world model | learned | off | off |
| P1P-C | learned world model | learned | learned | off |
| P1P-D | learned world model | learned | learned | learned |

P1P-A scores candidates from real ViZDoom branches. P1P-B scores the same
candidates through the frozen learned dynamics and learned reward head, without
terminal value or risk, then compares their predicted ordering with the actual
branch returns. B jointly measures dynamics and reward decision accuracy: a B
failure cannot establish which component caused it. C adds terminal value to B;
D adds risk to C. An additional intervention is necessary to separate a
dynamics failure from a reward failure. Do not label B's learned reward or the
recorded real returns as an oracle on imagined latent states.

The current ViZDoom `snapshot()/restore()` adapter reports
`snapshot_fidelity: world_state_only`: the native load does not rewind all game
time/reward accounting. Before using P1P-A branch returns, prove identical
observations, step rewards, terminal flags, and success flags by replaying each
fixed action prefix from a reset with the same seed. Stop if repeated branches
disagree. A world-state snapshot alone is insufficient evidence of identical
counterfactual starting states.

After P1D, check reset-based replay on the real host:

```bash
awa-v2-vizdoom-branch-replay \
  --output runs/v2_38_6/vizdoom_branch_replay.json
```

This probes two candidate continuations at both reset and a four-step prefix
on six fixed seeds. It rejects injected games and records all repeated trace
hashes. A passing probe qualifies only those probed prefixes; the full P1P
collector still has to check every state and candidate population it measures.

## Measurements to keep

For each fixed state, horizon, proposal type, and test, save candidate scores,
realized environment returns, action sequences, origin labels, chosen first
action, and measured planning latency. The gate computes Spearman rank
correlation and top-five agreement for predicted versus realized returns. It
also computes unique action sequences, normalized first-action entropy,
best-minus-median predicted score, candidate origin fractions, and the chosen
first-action distribution.

Run one closed-loop episode per fixed seed for each A-D planner variant at each
horizon and proposal type. Run a
matched random baseline and one P1P-A oracle-search episode per seed. Keep actor
success separate from planner success. Log actual realized return and success;
do not use training loss or predicted return as a replacement.

The raw artifact at `runs/v2_38_6/planner_diagnostic_raw.json` must have format
`awa-v2.38.6-planner-diagnostic-raw-v2`. The earlier v1 format is rejected
because its B condition required an impossible latent-state oracle. The v2 top level contains `scenario`,
`seed_ids`, the ordered `horizons`, world/actor `checkpoint_sha256`, `tests`,
`branch_replay_report_sha256` (the SHA-256 of `vizdoom_branch_replay.json`),
`random_baseline`, and P1P-A's `oracle_search_episode_results`. Each test contains
its declared dynamics/reward/value/risk settings, `episode_results`, and
`candidate_groups`. Each candidate group contains:

```text
seed, horizon, proposal, state_id
predicted_scores[], realized_returns[], action_sequences[], origins[]
chosen_first_action[], planning_latency_ms
```

Candidate groups for A-D must match in state ID, action sequences, origin labels,
and actual branch returns. A mismatch means the paired measurement is invalid.
Do not include `oracle_reward_provenance`: there is no oracle that scores the
checkpoint's imagined states. Episode rows contain `seed`, `horizon`,
`proposal`, boolean `success`, and `realized_return`.

The progressive gate recomputes the entire report from the adjacent
`planner_diagnostic_raw.json` and hashes the actual P1 world/actor checkpoint
files listed in the real structured-training receipt. A standalone PASS report
or a claimed checksum without those files and the matching real-host replay
qualification cannot advance P2.

Once the actual raw measurements exist, validate them with:

```bash
awa-v2-planner-diagnostic-gate \
  --config configs/v2_38_6_planner_diagnostic.yaml \
  --input runs/v2_38_6/planner_diagnostic_raw.json \
  --output runs/v2_38_6/planner_diagnostic.json
```

The gate requires at least six fixed seeds, all four tests, all horizons and
proposal populations, and complete paired candidate/episode measurements. It
reports a contiguous reliable joint dynamics/reward ranking horizon from P1P-B when Spearman is at
least 0.30 and top-five agreement at least 0.50. P2 is allowed only when the
complete diagnosis is present and P1P-A reaches at least 50% success while
beating the matched random baseline by at least 15 percentage points. A failure
keeps P2 blocked and points to search/objective setup before another training
campaign.
