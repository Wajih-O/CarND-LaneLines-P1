from typing import List, Tuple, Dict
import numpy as np
from dataclasses import dataclass

from helpers.point import Point

# TODO: Review/Complete Point class integration into the Segment implementation


class Segment:
    def __init__(self, segment: Tuple[int, int, int, int]):
        self.x1, self.y1, self.x2, self.y2 = segment
        # Sorted ends  with x (axis)  TODO:  refactor with Point class
        # if self.x2 < self.x1:
        #     self.x2, self.y2, self.x1, self.y1 = segment

    def __eq__(self, other):
        # TODO: sort the ends / support flipped ends
        return self.x1 == other.x1 and self.y1 == other.y1 and self.x2 == other.x2 and self.y2 == other.y2

    @property
    def from_(self) -> Point:
        return Point(self.x1, self.y1)

    @property
    def to_(self) -> Point:
        return Point(self.x2, self.y2)

    @property
    def ends(self) -> Dict[int, Point]:
        return  {0: Point(self.x1, self.y1), 1: Point(self.x2, self.y2)}

    @property
    def vect(self) -> Tuple[int, int]:
        """ Return vector (direction from point 1 to point 2 )"""
        return np.array([self.x2 - self.x1, self.y2 - self.y1])

    @property
    def slope(self) -> float:
        """ A helper to extract segment slope """
        return self.x2 - self.x1/(self.y2 - self.y1) # x = f(y)

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

    def point_distance(self, point: Point) -> float:
        """ Returns the  distance to the point resulting from the orthogonal projection """
        projected = self.project(point)
        return projected.euclidean_dist(point)

    def sim(self, other, similarity="cosine"):
        if similarity == "cosine":
            return self.cosine_sim(other)
        else:
            if similarity == "slope":
                return self.slope_exp_sim(other)
            raise Exception("Similarity not impelmented")

    def project(self, point: Point) -> Point:
        """ Project a point onto the segment"""
        segment =  Segment((self.x1, self.y1, point.x, point.y))
        # print("gen segment:", segment)
        projected_x, projected_y =   ((np.dot(self.vect,  segment.vect)/((self.norm)**2) * self.vect) + np.array([self.x1, self.y1])).ravel()
        return Point(int(projected_x), int(projected_y))

    def point_extend(self, point: Point, distance_threshold=20) -> "Segment":
        """ Extends the segment with a point (in the segment)/ if not -> with it projection"""
        projected = self.project(point) # it safer to project
        candidates = [Segment((self.x1, self.y1, projected.x, projected.y)),
                      Segment((projected.x, projected.y, self.x2, self.y2)),
                      self]
        return candidates[np.argmax(list(map(lambda x:x.norm,candidates)))]

    def match_ends(self, other:"Segment"):
        """ Match currend ends with  the other segment ends (as closest)"""
        # TODO: refactor the Segment class to use Point abstraction

        # Matching the first end
        indices = {0, 1}
        # first end matching
        matching_dict = {0: np.argmin([self.ends[0].euclidean_dist(other.ends[key]) for key in sorted(list(indices))])}
        indices.remove(matching_dict[0]) # removes the matching of the first item (index == 0)
        # the rest will be the match for the second end (index == 1)
        matching_dict.update({1: indices.pop()})
        assert len(indices) == 0
        return matching_dict

    def extend(self, other):
        """ Extend the current segment with the ends projection of the (take the longest)"""
        candidates = [self.point_extend(Point( other.x1, other.y1)),
                      self.point_extend(Point( other.x2, other.y2))]
        return candidates[np.argmax(list(map(lambda x:x.norm, candidates)))]

    def mutual_extend(self, other:"Segment", sort=False, reverse=True) -> List["Segment"]:
        """ Mutual extend both self with other and other with self """
        extended = [self.extend(other), other.extend(self)]
        if not sort:
            return extended
        return sorted(extended, key=lambda x: x.norm, reverse=reverse)


    def weighted_merge(self, other) -> "Segment":
        """ Weighted merge """
        # Extend each of the (self, other) and sort them decreasingly using their length/norm
        extended = self.mutual_extend(other, sort=True, reverse=True) # extended segments sorted by their norm
        matching_dict = extended[0].match_ends(extended[1])
        return matching_dict


    def merge(self, other:"Segment") -> "Segment":
        """ Merges/extends the bigger segment with the `smaller one` projection """
        # if other.norm > self.norm:
        #     return other.merge(self)
        # build the longest possible candidate to project onto the segment

        candidates = [self.extend(other), other.extend(self)]

        return candidates[np.argmax(list(map(lambda x:x.norm,candidates)))]

    def to_tuple(self):
        """ Transfrom segment to tuple """
        return self.x1, self.y1, self.x2, self.y2


def to_segments(items: List[Tuple[int, int, int, int]]) -> List[Segment]:
    """ A helper that transforms a list of tuples to a list of Segments (implementing helper methods)"""
    return sorted([Segment(item) for item in items], key= lambda segment: segment.slope)