# v2.38.4 cumulative training execution repair

The v2.38.3 P2 run collected 14,078 transitions and trained on a 10K replay selection, but constructed new belief encoder, world model, and actor weights. P1 used width 64 and P2 width 96. The P2 candidate scored 0/6 actor successes versus 2/6 for the P1 incumbent on the same navigation evaluation. This result does not establish the cause of failure, but the stage reset contradicted the cumulative training contract.

v2.38.4 resumes each next stage from the immediately preceding promoted world and actor checkpoints. It checks the registry hashes, track, observation/action dimensions, model width, horizons, and matching parent dataset hashes; it restores world optimizer and scaler states, world update count, actor and critic optimizer states, actor step count, and RNG states. Both structured and pixel tracks now use width 96 at every stage. A paired stage cannot qualify without hashes linking its resumed checkpoints to the prior promotion and a higher world update count. The frozen promotion thresholds and execution order are unchanged.

Start `runs/v2_38_4` from P0. The previous campaign cannot be resumed with these changes; retain its archive and hashes for comparison. Stop if P2 fails its paired gate. No architectural module was added.
