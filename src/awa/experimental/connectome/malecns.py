from __future__ import annotations
from pathlib import Path


class MaleCNSData:
    """
    Dataset locator only. This repository intentionally does not redistribute MaleCNS.
    Add official table-specific column mapping in a project-local adapter after download.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def require(self):
        if not self.root.exists():
            raise FileNotFoundError(
                f"MaleCNS data directory not found: {self.root}. "
                "Download the licensed/official dataset separately."
            )
        return self.root
