// Aether v2.14 Unity reference contract (protocol sketch, not a drop-in networking package).
// Mirror the JSONL operations documented in docs/GAME_BRIDGE_PROTOCOL.md.
// Keep game-state serialization deterministic if snapshot/restore is enabled.
using UnityEngine;

public interface IAetherGameEnvironment
{
    float[] ResetEnvironment(int? seed);
    float[] GoalVector();
    AetherStep Step(float[] action);
    byte[] Snapshot();
    float[] Restore(byte[] snapshot);
}

public struct AetherStep
{
    public float[] observation;
    public float reward;
    public bool terminated;
    public bool truncated;
    public string infoJson;
}
