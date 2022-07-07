from ..builder import PIPELINES
from typing import Optional


@PIPELINES.register()
class Mosaic:
    def __init__(
        self,
        neg_dir: Optional[str] = None,
    ) -> None:
        self.neg_dir = neg_dir


@PIPELINES.register()
class FilterAnnotations:
    def __init__(self) -> None:
        pass
