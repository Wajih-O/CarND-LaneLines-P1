from typing import List, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class Point:
    """ Point class to represent a pixel coordinate """
    # TODO: integrate the point to Segment implementation as it provides better readability
    # for merge
    x:int
    y:int


class Segment:
    def __init__(self, segment: Tuple[int, int, int, int]):
        self.x1, self.y1, self.x2, self.y2 = segment
        self.__similarities = {"cosine"}
    @property
    def vect(self) -> Tuple[int, int]:
        """ det direction vector """
        return np.array([self.x2 - self.x1, self.y2 - self.y1])
    @property
    def slope(self) -> float:
        """ A helper to extract segment slope"""
        return self.x2 - self.x1/(self.y2 - self.y1) # slope height = f(width)

    @property
    def norm(self):
        """ A helper to extract segment length """
        return np.linalg.norm([self.x2 - self.x1, self.y2 - self.y1])

    @property
    def length(self):
        return self.norm

    def __repr__(self) -> str:
        return f'({self.x1}, {self.y1}) <-> ({self.x2}, {self.y2})'


    # Similarity measures
    def slope_exp_sim(self, other: "Segment", sigma=10):
        """ A similarity measurement base on the  """
        return np.exp(-np.abs(self.slope - other.slope)/sigma)

    def cosine_sim(self, other: "Segment"):
        """ A cosine similarity """
        return  np.dot(self.vect,  other.vect)/(self.norm*other.norm)

    def sim(self, other, similarity="cosine"):
        if similarity == "cosine":
            return self.cosine_sim(other)
        else:
            raise Exception("Similarity not impelmented")

    def project(self, point: Point) -> Point:
        """ Project a point onto the segment"""
        segment =  Segment((self.x1, self.y1, point.x, point.y))
        # print("gen segment:", segment)
        projected_x, projected_y =   ((np.dot(self.vect,  segment.vect)/((self.norm)**2) * self.vect) + np.array([self.x1, self.y1])).ravel()
        return Point(int(projected_x), int(projected_y))

    def extend(self, point: Point) -> "Segment":
        """ Extends the segment with point (in the segment) if not with it projection"""
        projected = self.project(point) # it safer to project
        candidates = [Segment((self.x1, self.y1, projected.x, projected.y)),
                      Segment((projected.x, projected.y, self.x2, self.y2)),
                      self]
        return candidates[np.argmax(list(map(lambda x:x.norm,candidates)))]


    def merge(self, other:"Segment") -> "Segment":
        """ Merges the current segment with the `other` segment """
        # Todo : project the smaller segment (using norm) onto, the bigger one
        if other.norm > self.norm:
            return other.merge(self)
        # build the longest possible candidate to project onto the segment
        candidates = [self.extend(Point( other.x1, other.y1)),# Segment(self.x1, self.y1, self.x2, other.y2))
                      self.extend(Point( other.x2, other.y2))]
        return candidates[np.argmax(list(map(lambda x:x.norm,candidates)))]

    def to_tuple(self):
        """ Transfrom segment to tuple """
        return self.x1, self.y1, self.x2, self.y2


def to_segments(items: List[Tuple[int, int, int, int]]) -> List[Segment]:
    """ A helper that transforms a list of tuples to a list of Segments (implementing helper methods)"""
    return sorted([Segment(item) for item in items], key= lambda segment: segment.slope)