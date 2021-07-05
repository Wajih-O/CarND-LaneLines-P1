from typing import List, Tuple, Optional
import math
import numpy as np
from itertools import product
from queue import PriorityQueue
from abc import abstractmethod

from itertools import chain

from pprint import pprint
import logging

import cv2
from segment import Segment

# Configure logging should be moved
logging.basicConfig(filename='utils.log', level=logging.DEBUG)



def canny(gl_image, low_threshold=50, high_threshold=150):
    """Applies the Canny transform"""
    return cv2.Canny(gl_image, low_threshold, high_threshold)

def gaussian_blur(gl_image, kernel_size=5):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(gl_image, (kernel_size, kernel_size), 0)



class Mergeable:
    """ A helper class to contain merged nodes indexed by the sorted tuple of original indices"""
    def __init__(self, id_:Tuple[int]):
        self._id = id_ # tuple(sorted(list(id_))) # tuple merged  (the order matters)

    def __repr__(self):
        return str(self._id)

    def __hash__(self):
        return int(''.join(map(str,sorted(self._id))))

    def cache(self, store:dict):
        """ Compute the merge and store it """
    def retrieve(self, store) -> Optional["Merge"]:
        """Retrieves the merge from a store"""
        return store.get(self._id, None)

    @abstractmethod
    def compute(self, cache: Optional[dict]=None, update=False):
        """ Compute the merge using optionally a store/cache"""


class SegmentCluster(Mergeable):
    def __init__(self, id_:Tuple[int], data=List[Segment]):
        super().__init__(id_)
        self.__data = data
        self._logger = logging.getLogger(__name__) # init logger


    def compute(self, cache: Optional[dict]=None, update=False):
        """ Compute a segment merge from original container and a cache
            :cache: a cache/storing precomputed merge
            :param update: update the cache with the newly computed items.
        """
        self._logger.debug(self._id)
        if cache and self._id in cache:
            return cache[self._id]

        # merge
        if len(self._id) == 1:
            return self.__data[self._id[0]]
        output = SegmentCluster(self._id[:-1], data=self.__data).compute(cache, update=update).merge(self.__data[self._id[-1]])
        if update:
            cache[self._id] = output
        return output

    def sim(self, other: "SegmentCluster", similarity="cosine", merge=True, linkage=np.min, cache:Optional[dict]={}):
        """ A helper to compute similarity between two Segment clusters
            using element-wise-similarity/or merged
            :param other: The segment cluster with which we want compute the similarity
            :param merge: Merge the cluster to one segment before
            :param linkage: The type of aggregation of individual similarity.
            :param cache:
        """
        if merge:
            # Merge/interpolate/average the segment cluster into one segment before applying the similarity
            return self.compute().sim(other.compute(),similarity=similarity)

        else:
            # TODO: (optionally) use the linkage function to compute similarity between the two cluster instead of the merge/extended segment
            pass


class SimContainer:
    """ A helper class that contains similar items for fast retrieval"""
    def __init__(self, item):
        self.__item = item
        self.__queue = PriorityQueue()
        self.__best = None

        self._similar = set() # keeps track of similar items
        self._removed = set() # keeps track of  the removed keys/items

    def __repr__(self):
        repr_str = ''
        if self.__best:
            repr_str += str(self.__best)
        repr_str += str(self.__queue.queue)
        return repr_str

    def put(self, similar, similarity):
        """ Put a similar item in the container
        :param similar: a similar item (hashable) or a key
        :param similarity: a non zero similarity to the container self._item
        """
        if similar in self._removed:
            self._removed.remove(similar)
        self._similar.add(similar)

        self.__queue.put((1/similarity, similar, similarity))
        if self.__best:
            self.__queue.put(self.__best) # put back the best item to compete with the new one

        self.__best = self.__queue.get() # caching the best similar item (according to similarity)
        while self.__best[1] in self._removed: # the best is not valid anymore as it was removed
            self.__best = self.__queue.get()

    def remove(self, key):
        """ A helper to remove an item from the container (only by key)"""
        if key in self._similar:
            # the item exist -> update the queue
            self._similar.remove(key)
            self._removed.add(key)
            while self.__best and self.__best[1] in self._removed:
                # update the best
                if not self.__queue.empty():
                    self.__best = self.__queue.get()
                else:
                    self.__best = None

    def best(self):
        """ return the best similarity in this container"""
        return self.__best

    @property
    def queue(self):
        """ return the queue. """
        return self.__queue.queue
    @property
    def similar(self):
        similar_items = []
        if self.__best is None:
            return []
        else:
            similar_items.append(self.__best[1]) # update the similar items with the best one (cached)
        if len(self.__queue.queue):
            similar_items.extend([item[1] for item in self.__queue.queue if item[1] not in self._removed])
        return similar_items


def agglomerate(data: List[Segment], threshold: float, similarity="cosine") -> Tuple[dict, dict]:
    """ Agglomerate data item given
        using agglomerative-clustering (grouping the closest cluster first)

    (todo: simultaneous grouping before updating the)

    :param threshold: similarity threshold above which 2 items
                     are considered similar and could be merged
    :param similarity: The similarity measure between 2 segments

    :return: Tuple/Couple of clusters' dictionary where the keys are the cluster identified by the index of the item in the original data
    the `data` parameter and the values are the similar item non yet merged/will not because the similarity with these items is below the
    `threshold` parameter. And the store of computed merged segment of the cluster and the intermediate merges that lead to the final
    returned clusters.
    """
    logger = logging.getLogger(__name__)
    def get_merge_candidates(sim_dict):
        """ a helper to get candidates to merge"""
        pq_ = PriorityQueue()
        for key, value in sim_dict.items():
            best = value.best() # get best item from the SimContainer (value)
            if best:
                pq_.put((best, key))

        if  pq_.empty():
            return None, None
        else:
            return pq_.get()


    # Build the similarity between
    data_dict = {} # a dictionary
    for index, item in  enumerate(data):
        data_dict[(index,)] = item

    # Sorted segment by length (from the longest to the shortest segment)
    sorted_segments = sorted([(key, segment) for key, segment in data_dict.items()], key=lambda x: x[1].norm, reverse=True)

    # populate similarities (Matrix as a dictionary)
    similarities = {}
    for i in range(len(sorted_segments)):
        # similarities.
        key_i, segment_i = sorted_segments[i]
        similarities[key_i] = SimContainer(segment_i)
        for  j in range(i+1, len(sorted_segments)):
            key_j, segment_j = sorted_segments[j]
            similarities[key_i].put(key_j, segment_i.sim(segment_j))

    # Search for the best items to agglomerate/merge
    best, orig  = get_merge_candidates(similarities)
    while best and (best[2] > threshold):
        to_merge = {orig, best[1]}
        logger.debug(f"  Merging (clusters) -> {to_merge}")
        merged = SegmentCluster(orig+best[1], data) # Merged cluster
        merged_segment = merged.compute(cache=data_dict, update=True)

        # Similar candidates to the cluster to update their similarity entries (previous similar to the (orig and best))

        to_update = set(chain(*[container.similar for container in
            filter(lambda item: item is not None, [similarities.get(key, None) for key in  to_merge])]))

        # remove the merged part
        for key in to_merge:
            if key in similarities:
                del similarities[key]

        # remove the merged item from the other similarity container
        for key, sim_container in similarities.items():
            sim_container.remove(orig)
            sim_container.remove(best[1])

        to_update.difference_update({orig, best[1]}) # clean to_update from the already merged items
        to_update.update(similarities.keys())

        similarities[merged._id] = SimContainer(merged_segment)

        for key in to_update:
            similarities[merged._id].put(key, merged_segment.sim(data_dict[key]) )
        # get the next best candidates to agglomerate
        best, orig  =  get_merge_candidates(similarities)
        # pprint(similarities)

    return similarities, data_dict
