# v2.38.3: bounded replay selection correction

The original v2.38.0 release is preserved. This version changes one execution path: how a later ViZDoom campaign stage materializes its fixed transition budget from append-only replay. The promotion criteria and model architecture are unchanged.

## Grounded P2 failure

On the v2.38.2 RunPod campaign, P2 collected 14,604 total replay transitions. Its 10,000-transition training file took the earliest prefix and contained **zero** successful `my_way_home` episodes. The last replay shard had three successful episodes; all were past that cutoff. P2's candidate consequently failed the paired success gate (actor 0/6 versus incumbent 2/6). The comparison still stands as the actual v2.38.2 result.

## Correction

For later stages, the materializer keeps every transition in the first completed P1 collection and takes the most recent transitions to fill the frozen stage budget. It selects by replay position, without inspecting reward, success, policy, or evaluation. It requires the retained first collection to end at a true terminal and checks the resulting size. P1's existing materialization and all general replay call sites keep their prior behavior.

Applying the corrected selection to the existing replay yields exactly 10,000 rows; all three observed P2 successful episodes occur in those rows. This only confirms that the learner *can receive* the successful experience. It does not establish policy learning.

## Qualification

Use a new `runs/v2_38_3` evidence root with the v2.38.3 archive SHA-256 and unchanged promotion gates. Re-run P0, P1, and P2 on a sufficiently sized GPU volume. Preserve the v2.38.2 failed receipt and the replay hashes for comparison. Examine action mapping, rewards, actual episode success and full paired held-out evaluation. Do not progress to P3 unless P2 passes the frozen gate. Expand the idle 10 GB evidence-only network volume before bootstrapping training: the v2.38.3 preflight retains a 100 GiB free-space reserve.
