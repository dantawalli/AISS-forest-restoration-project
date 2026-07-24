from abc import ABC, abstractmethod


class BaseContextBuilder(ABC):
    """
    Base class for all Context Builders in FYNOS AI.

    Every Context Builder transforms a Source of Truth
    into a standardized AI Context Object.
    """

    CONTEXT_VERSION = "1.0"

    @abstractmethod
    def build(self, source_data: dict) -> dict:
        """
        Build a standardized Context Object.

        Parameters
        ----------
        source_data : dict
            Source of Truth object.

        Returns
        -------
        dict
            Standardized context object.
        """
        pass