"""Deprecated compatibility shim for the pre-v2.30 hierarchy prototype."""
from __future__ import annotations
import warnings
warnings.warn(
    "awa.v2.hierarchy is outside the v2.31 stable execution spine; import from "
    "awa.experimental.legacy_v2.hierarchy only for historical reproduction.",
    DeprecationWarning,
    stacklevel=2,
)
from awa.experimental.legacy_v2.hierarchy import *  # noqa: F401,F403
