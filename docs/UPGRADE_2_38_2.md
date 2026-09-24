# v2.38.2 execution correction

This release corrects execution and evidence handling only. The action-time `agent_runtime` tree remains byte-for-byte unchanged. Preserve the original v2.38.0 ZIP, its hashes, and `runs/v2_38/` as historical evidence; do not edit a P2 receipt to make it pass.

## Corrected decisions

- Each stage evaluates its incumbent and candidate on that stage's scenario, episode count, and reset seeds before applying the frozen regression margins. The stage result records the paired evaluation and comparison contract.
- P1 requires the promoted P1 checkpoint. P2, P3, and pixel P8 additionally require their exact stage to be promoted, a recorded paired comparison, and at least one successful evaluation episode across actor or planner. Zero success fails this minimal operational gate; a success on six episodes is **not** evidence of statistically established learning. This operational floor was added after observing v2.38.0 P2 and applies prospectively to fresh v2.38.1+ data; it does not turn the old 0/6 result into evidence of learning.
- An executor command that exits cleanly but does not meet its evidence gate now records `FAILED` and exits with code 2.
- A cached v2.38.0 P2/P3 verdict is marked `MANUAL_REQUIRED` for a fresh campaign directory; an old ledger entry cannot silently reuse its cross-scenario decision.
- P9 accepts only a promoted P3 `real-control-25k` structured checkpoint whose actor and world files match the receipt's SHA-256 hashes.
- Pixel V-JEPA precomputation now flushes bounded clip batches instead of retaining every 64-frame clip in host RAM. The shipped pixel batch size is 1 to keep inference memory bounded on the 24 GB RTX 4090 class host.

## Restart the affected campaign

Install this build on the CUDA host and run `bash deploy/runpod/bootstrap.sh`. Use the new evidence root `runs/v2_38_2`; keep `runs/v2_38` archived. The executor will start with P0 and proceed one phase at a time:

```bash
awa-v2-progressive-executor --config configs/v2_38_progressive_executor.yaml --evidence-root runs/v2_38_2
awa-v2-progressive-executor --config configs/v2_38_progressive_executor.yaml --evidence-root runs/v2_38_2 --execute-next
```

The previous P2 checkpoint had zero actor and planner successes on `my_way_home` and was rejected. Do not carry it forward as qualified training. If the new P2 still has zero success or fails a same-scenario regression comparison, stop and investigate the learning setup using the actual rollout and loss evidence. The full seven-system matrix remains mandatory before P15 scaling authorization.
