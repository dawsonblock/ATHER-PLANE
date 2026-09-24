"""Deprecated compatibility shim for the pre-v2.30 decision prototype."""
from __future__ import annotations
import warnings
warnings.warn(
    "awa.v2.decision is outside the v2.31 stable execution spine; import from "
    "awa.experimental.legacy_v2.decision only for historical reproduction.",
    DeprecationWarning,
    stacklevel=2,
)
from awa.experimental.legacy_v2.decision import *  # noqa: F401,F403
