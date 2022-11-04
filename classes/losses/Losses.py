from abc import ABC, abstractmethod

class Losses(ABC):
    """
    Abstract class for losses
    """
    ## PROPERTIES
    
    # NO PROPERTIES

    ## METHODS
    @classmethod
    @abstractmethod
    def calculate(self, target, prediction) -> float:
        """
        Calculate the losses for the given prediction(s) against target(s)
        """
        pass