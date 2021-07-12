from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass


from lane_detection.point import Point

# TODO: Complete Point class integration into the Segment implementation


class Segment:
    def __init__(self, segment: Tuple[int, int, int, int]):
        self.x1, self.y1, self.x2, self.y2 = segment
        # Sorted ends  with x (axis)  TODO:  refactor with Point class
        # if self.x2 > self.x1:
        #     self.x2, self.y2, self.x1, self.y1 = segment

    def __eq__(self, other):
        # TODO: sort the ends / support flipped ends
        return self.x1 == other.x1 and self.y1 == other.y1 and self.x2 == other.x2 and self.y2 == other.y2

    @property
    def equation(self) -> Tuple[float, float, float]:
        """ returns line equation ax + by + c = 0"""
        if self.horizontal:
            return (0, 1, -self.y1)
        if self.vertical:
            return (1, 0, -self.x1)
        else:
            return (-self.slope,  1,  (self.slope * self.x1) - self.y1)

    def intersection(self, other) -> Optional[Point]:
        """" Compute intersection using homogenous coordinates """
        a1, b1, c1 = self.equation
        a2, b2, c2 = other.equation
        c_inter = a1*b2 - a2 *b1
        if c_inter:
            return Point(int((b1*c2 - b2*c1)/c_inter) , int((a2*c1 - a1*c2)/c_inter))
        return None

    @property
    def horizontal(self) -> bool:
        return self.y1 == self.y2

    @property
    def vertical(self) -> bool:
        return self.x1 == self.x2

    def nearly_horizontal(self, threshold=.2):
        if self.horizontal:
            return True
        return np.abs(self.slope) < threshold

    def nearly_vertical(self, threshold=.1):
        return self.vertical or (np.abs(np.dot(self.vect, np.array((1, 0))))/self.norm < threshold)

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
        if self.vertical:
            raise Exception("Vertical segment/line !")
        return (self.y2 - self.y1)/(self.x2 - self.x1) # height = f(width)

    @property
    def norm(self):
        """ A helper to extract segment length """
        return np.linalg.norm([self.x2 - self.x1, self.y2 - self.y1])

    @property
    def length(self):
        return self.norm

    def __repr__(self) -> str:
        return f'({self.x1}, {self.y1}) -> ({self.x2}, {self.y2})'

    # Similarity measures
    def slope_exp_sim(self, other: "Segment", sigma=1):
        """ A similarity measurement base on the  """
        return np.exp(-np.abs(self.slope - other.slope)/sigma)

    def cosine_sim(self, other: "Segment"):
        """ A cosine similarity """
        return  np.dot(self.vect,  other.vect)/(self.norm*other.norm)

    def cross(self, other:"Segment") -> List[Tuple["Segment", "Segment"]]:
        """ Combine """
        return [ (self, other),
                 (Segment(self.ends[0].as_tuple + other.ends[0].as_tuple), Segment(self.ends[1].as_tuple + other.ends[1].as_tuple)),
                 (Segment(self.ends[0].as_tuple + other.ends[1].as_tuple), Segment(self.ends[0].as_tuple + other.ends[1].as_tuple))]


    def min_config_cosine(self, other: "Segment"):
        return np.mean([np.absolute(segment1.cosine_sim(segment2)) for segment1, segment2 in self.cross(other)])


    def point_distance(self, point: Point) -> float:
        """ Returns the  distance to the point resulting from the orthogonal projection """
        projected = self.project(point)
        return projected.euclidean_dist(point)

    def sim(self, other, similarity="cosine"):
        if similarity == "cosine":
            return self.min_config_cosine(other)
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

    def point_extend(self, point: Point, distance_threshold=20, inclusive=True) -> "Segment":
        """ Extends the segment with a point (in the segment)/ if not -> with it projection"""
        projected = self.project(point) # it safer to project
        candidates = [Segment((self.x1, self.y1, projected.x, projected.y)),
                      Segment((projected.x, projected.y, self.x2, self.y2))]
        if inclusive:
            candidates.append
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
        """ Weighted merge (interpolate segment) risky if the 2 segments  """
        # Extend each of the (self, other) and sort them decreasingly using their length/norm
        extended = self.mutual_extend(other, sort=True, reverse=True) # extended segments sorted by their norm
        matching = extended[0].match_ends(extended[1])

        match_1 = Segment(extended[0].ends[0].as_tuple + extended[1].ends[matching[0]].as_tuple)
        match_2 = Segment(extended[0].ends[1].as_tuple + extended[1].ends[matching[1]].as_tuple)


        weight = extended[1].norm/( extended[1].norm + extended[0].norm)
        t1_x, t1_y = (match_1.vect*weight).ravel()
        t2_x, t2_y = (match_2.vect*weight).ravel()

        # Translate the longer extended segments ends to their final position
        end_0_translated = Segment(extended[0].ends[0].as_tuple + extended[0].ends[0].translate(Point(t1_x, t1_y)).as_tuple)
        end_1_translated = Segment(extended[0].ends[1].as_tuple + extended[0].ends[1].translate(Point(t2_x, t2_y)).as_tuple)

        return Segment(end_0_translated.ends[1].as_tuple + end_1_translated.ends[1].as_tuple)



    def merge(self, other:"Segment") -> "Segment":
        """ Merges/extends the bigger segment with the `smaller one` projection """
        # return self.weighted_merge(other) # alternatively run a weighted merge to clean-up
        # extends the longer segment with the ends of smaller one alternatively run a weighted merge to clean-up

        # TODO: Make this mothod configurable
        # Config 1:
        if self.norm > other.norm:
            return self.extend(other)
        else:
            return other.extend(self)

        # Mutual extend
        # candidates = self.mutual_extend(other)
        # return candidates[np.argmax(list(map(lambda x:x.norm,candidates)))]

    def to_tuple(self):
        """ Transfrom segment to tuple """
        return self.x1, self.y1, self.x2, self.y2


def to_segments(items: List[Tuple[int, int, int, int]]) -> List[Segment]:
    """ A helper that transforms a list of tuples to a list of Segments (implementing helper methods)"""
    return sorted([Segment(item) for item in items], key= lambda segment: segment.slope)