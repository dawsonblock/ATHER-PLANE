# Remaining work after v2.35

1. Run `awa-v2-execution-preflight` on the actual rented GPU host and preserve
   the resulting `execution_preflight.json` with the experiment artifacts.
2. Execute `configs/v2_35_vizdoom_structured.yaml` through
   `awa-v2-vizdoom-real-campaign`. Fix only failures exposed by the native Doom
   execution path before scaling architecture or data volume.
3. After structured control is stable, execute the 1K→5K pixel/V-JEPA bring-up.
   Do not jump directly to a large RGB campaign.
4. Run the planner schedule sweep `[0, 32, 8, 4, 1]` on the actual target GPU
   using a frozen world/actor checkpoint. Require matched logical work and
   numerical action equivalence before comparing latency/VRAM.
5. Execute the v2.34 five-seed fixed-vs-DREAM-RSI experiment unchanged. The
   paper's results motivate the experiment but are not Aether embodied-RL data.
6. Execute the independent seven-system 25K/100K ablation. DREAM-RSI and the
   seven-system ladder answer different causal questions.
7. Once real ViZDoom evidence exists, compare the surviving Aether runtime with
   at least one serious external model-based RL baseline under matched
   transition and accelerator budgets.
8. Keep energy reporting `not_measured` until a direct hardware-supported energy
   counter is available. Do not substitute TDP × wall time.
9. Do not add new cognitive modules until the real structured/pixel bring-up and
   first multi-seed experiments reveal an actual missing capability.
