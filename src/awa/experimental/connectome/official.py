from __future__ import annotations
from pathlib import Path
import csv
from awa.experimental.connectome.graph import compile_edge_csv

SOURCE_ALIASES = ("source", "pre", "bodyId_pre", "body_id_pre", "pre_root_id", "source_id")
TARGET_ALIASES = ("target", "post", "bodyId_post", "body_id_post", "post_root_id", "target_id")
WEIGHT_ALIASES = ("weight", "syn_count", "synapse_count", "count", "n_synapses")


def _pick(fields, aliases, kind):
    for name in aliases:
        if name in fields: return name
    raise ValueError(f"Could not identify {kind} column. Fields={fields}")


def detect_connection_columns(path: str | Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        fields = next(reader)
    return {
        "source": _pick(fields, SOURCE_ALIASES, "source"),
        "target": _pick(fields, TARGET_ALIASES, "target"),
        "weight": _pick(fields, WEIGHT_ALIASES, "weight"),
    }


def compile_official_connections(path: str | Path):
    cols = detect_connection_columns(path)
    return compile_edge_csv(path, cols["source"], cols["target"], cols["weight"])
