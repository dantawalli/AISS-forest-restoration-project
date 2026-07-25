from abc import ABC, abstractmethod


class BaseReasoningContextBuilder(ABC):
    """Base class for all reasoning context builders."""

    @abstractmethod
    def build(self, context_data: dict) -> dict:
        pass