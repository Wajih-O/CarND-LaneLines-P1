from typing import List, Tuple, Optional, Dict
import math
import numpy as np
from itertools import product
from queue import PriorityQueue
from abc import abstractmethod

from itertools import chain, combinations

from pprint import pprint
import logging

import cv2

from lane_detection import Segment, Point
from lane_detection.hough_lines import HoughLines
from lane_detection.clustering import agglomerate
from lane_detection.visu_utils import draw_lane


# Configure logging should be moved
logging.basicConfig(filename="utils.log", level=logging.DEBUG)


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def hsv_value(img):
    """ " Convert img to HSV and extract the value channel"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]


def canny(gl_image, low_threshold=50, high_threshold=150):
    """Applies the Canny transform"""
    return cv2.Canny(gl_image, low_threshold, high_threshold)


def gaussian_blur(gl_image, kernel_size=5):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(gl_image, (kernel_size, kernel_size), 0)


def horizon_crop(image: np.ndarray, horizon: Optional[int] = None) -> np.ndarray:
    """A helper to remove part of image above horizon line"""
    output_image = image.copy()
    if horizon is None:
        # assume the camera plan is normal to the road (the camera not tilted)
        output_image[
            : image.shape[0] // 2, :
        ] = 0  # to keep it compatible with color image
    return output_image


def weighted_img(img, initial_img, α=0.8, β=1.0, γ=0.0):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ

    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def interest_region_crop(image):
    """A helper masking the image with Mask with region of interest"""
    if len(image.shape) == 3:
        # multi-channel image (color)
        height, width, _ = image.shape
    else:
        height, width = image.shape

    left_bottom = [0, height - 1]
    right_bottom = [width, height - 1]
    apex = [width // 2, int(height / 1.7)]

    interest_region_polygon = np.array(
        [list(map(lambda xy: (xy[0], xy[1]), [left_bottom, apex, right_bottom]))]
    )
    masked_edges = np.copy(image)
    inv_mask = np.copy(image)
    cv2.fillPoly(inv_mask, pts=interest_region_polygon, color=(0))
    return masked_edges - inv_mask


# for slope Similarity it is a exp kernel on absolute difference with configurable sigma (for the kernel)
config = {"threshold": 0.98, "similarity": "slope", "slope_threshold": 0.5}


def extract_lane(image, config=config) -> List[Segment]:
    """A helper to extract the  2 segments defining the lane"""
    # Detect edge using canny on a gaussian blur
    if len(image.shape) not in {2, 3}:
        raise Exception("Not supported dimension")
    if len(image.shape) == 3:
        width, height, _ = image.shape
    else:
        assert len(image.shape) == 2
        width, height = image.shape
    logging.debug(f"image dimension {width}x{height}")
    edges_image = interest_region_crop(
        horizon_crop(canny(gaussian_blur(grayscale(image))))
    )
    # Extract segments (detect lines) then filter the horizontal and nearly horizontal ones
    segments = list(
        filter(
            lambda segment: not segment.nearly_vertical(0.2)
            and not segment.nearly_horizontal(config.get("slope_threshold", 0.5)),
            [Segment(item) for item in np.squeeze(HoughLines()(edges_image))],
        )
    )
    # Iterative segments merge (agglomerative approach with shrinking threshold)

    threshold = config.get("threshold")
    while len(segments) > 2:  # iterative adaptation of the threshold
        cluster_dict, segments_store = agglomerate(
            segments, threshold=threshold, similarity=config.get("similarity")
        )
        # Filter/Sort and Select the 2 longest (merged/extended line) segments as the candidate for lane detection
        merged_segments = sorted(
            filter(
                lambda segment: segment is not None,
                [segments_store.get(key, None) for key in cluster_dict],
            ),
            key=lambda segment: segment.length,
            reverse=True,
        )
        segments = merged_segments[:]
        threshold *= 0.9  #  more tolerance to the similarity threshold

    right, left = None, None
    left_right = sorted(
        sorted(
            filter(lambda segment: segment.length > 40, segments),
            key=lambda segment: segment.length,
            reverse=True,
        ),
        key=lambda segment: segment.slope,
    )[:2]
    if len(left_right) == 2:
        left, right = left_right
        # extending the right and left side with the projection of the center of the image
        # expectedly close to the horizon (to replace with the intersection of the 2 lane lines/sides)
        height, width = image.shape[:2]
        # center = Point(width//2, height//2)
        # Extending the right and left side segments of the lane to their intersection
        intersection = right.intersection(left)
        right = right.point_extend(intersection, inclusive=False)
        left = left.point_extend(intersection, inclusive=False)
        # Extending the right and left side segments of the lane to the bottom of the image
        right = right.point_extend(
            right.intersection(Segment((0, height - 1, width - 1, height - 1))),
            inclusive=False,
        )
        left = left.point_extend(
            left.intersection(Segment((0, height - 1, width - 1, height - 1))),
            inclusive=False,
        )

    # preparing output dict with left and right annotation as keys
    output_dict = {}
    if left:
        output_dict["left"] = left
    if right:
        output_dict["right"] = right

    return output_dict


def save_image(image, output_path: str):
    """Save/Write image to output_path"""


def find_lane(image) -> List[Segment]:
    extraction_output = extract_lane(image)
    draw_lane(image, extraction_output)
    return extraction_output
