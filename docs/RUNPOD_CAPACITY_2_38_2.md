# RunPod capacity plan for the full v2.38.2 campaign

## Provisioning target

Keep the existing 200 GB persistent network volume if it has at least 100 GiB free after the repository, Python environment, and V-JEPA weights are installed. If it does not, increase that volume to **300 GB** in the RunPod Storage page. RunPod allows network volume size increases but does not allow shrinking; the documented standard-storage price is $0.07/GB/month for the first TB, so 300 GB is about $21 USD/month before tax. Check the current console price before changing capacity.

Attach the volume to the pod in the same data center, mounted at /workspace. Use the standard tier unless actual I/O measurements show checkpointing or dataset reads are the bottleneck. The campaign is sequential and the datasets/checkpoints are small; extra IOPS will not fix GPU compute time or V-JEPA memory use.

RunPod network volume capacity is configured in the console and may not be visible as a reliable quota through statvfs inside the container. Confirm the configured GB capacity in the console. On the pod, check free space and the largest directories with:

    df -h /workspace
    du -sh /workspace/*

The v2.38.2 bootstrap and P0 preflight require 100 GiB free on the mounted workspace. The bootstrap checks before creating the virtual environment, avoiding a half-installed environment when the disk is already low.

## Why 300 GB is a safe target

The actual archived P0–P2 evidence contains 7.54 MiB uncompressed. Its 10K structured replay materializes to about 0.53 MiB, and individual training checkpoints are around 1–1.5 MiB. This is not a good proxy for RGB data, so reserve more space for the pixel path:

- The 5K RGB track uses 320×240 RGB frames. Raw current and next observations alone have an upper bound of about 2.2 GiB before compression; the append-only shards, staged dataset copies, and feature dataset add copies.
- Reserve 20 GiB for the RGB replay, V-JEPA cache, intermediate files, and any compression expansion.
- Reserve 20 GiB for the Python/Torch environment, package caches, and V-JEPA model/processor files.
- Reserve 40 GiB for all campaign checkpoints, receipts, evaluation data, and recoverable evidence snapshots.
- Keep at least 100 GiB free before starting. This leaves room for later cumulative 250K/500K/1M artifacts and recovery from interrupted runs.

Those are conservative operational allowances, not predicted consumption. Delete pip/temporary caches only after confirming they can be recreated. Never delete replay shards, checkpoints, receipts, or reports to make space.

## Host memory and V-JEPA

Disk is not the main P8 risk. A 64-frame 320×240 RGB clip is about 14.1 MiB as uint8; keeping 10,000 clips pending would require roughly 138 GiB just for the clips. The previous precompute implementation retained that entire pending set. v2.38.2 now encodes and writes a bounded batch before reading more clips, and the pixel config uses batch size 1. The source replay arrays still occupy host RAM, so use a pod with at least 32 GiB system RAM; 64 GiB is a safer target.

## Storage placement

- Network volume (/workspace): repository, Python environment, Hugging Face cache, runs/v2_38_2, checkpoints, and reports.
- Container disk: operating system and disposable temporary files only. It is erased when the pod stops.
- Evidence backup: periodically copy runs/v2_38_2 to a second location outside the pod. The network volume is persistent storage, not a second independent backup.

Do not use the full seven-system matrix as the first capacity test. Complete P0–P3, inspect the measured directory totals, back up the evidence, and then continue in executor order.
