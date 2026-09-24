"""Deprecated compatibility shim for the pre-v2.30 context_replay prototype."""
from __future__ import annotations
import warnings
warnings.warn(
    "awa.v2.context_replay is outside the v2.31 stable execution spine; import from "
    "awa.experimental.legacy_v2.context_replay only for historical reproduction.",
    DeprecationWarning,
    stacklevel=2,
)
from awa.experimental.legacy_v2.context_replay import *  # noqa: F401,F403
