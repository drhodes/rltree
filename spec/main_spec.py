"""
main spec
"""

# pyrefly: ignore [missing-import]
from libspec import Spec

from . import app, code_quality, rltree


class MainSpec(Spec):
    def modules(self) -> list[object]:
        return [app, rltree, code_quality]
