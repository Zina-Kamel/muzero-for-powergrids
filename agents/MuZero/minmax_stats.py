import collections
from typing import Optional

# from MuZero Paper
MAXIMUM_FLOAT_VALUE = float('inf')

# KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

class MinMaxStats(object):
    def __init__(self, min, max):
        
        self.maximum = max if max else -MAXIMUM_FLOAT_VALUE
        self.minimum = min if min else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value