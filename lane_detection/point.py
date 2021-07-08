from typing import List, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class Point:
    """ Point class to represent a pixel coordinate """

    x:int
    y:int

    def euclidean_dist(self, other:"Point"):
        return np.linalg.norm([other.x-self.x, other.y-self.y])

    @property
    def as_tuple(self):
        return (self.x, self.y)

    def translate(self, t_: "Point"):
        """ Translate the point by vector `t_` (as Point) """
        return Point(x=int(self.x + t_.x), y= int(self.y + t_.y))