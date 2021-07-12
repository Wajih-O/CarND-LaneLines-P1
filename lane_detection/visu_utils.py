from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from lane_detection.segment import Segment


def draw_lines(img, lines: List[Segment], color=[255, 0, 0], thickness=4):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image in place (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    for segment in filter(lambda x: x is not None, lines):
        cv2.line(
            img, (segment.x1, segment.y1), (segment.x2, segment.y2), color, thickness
        )
    return img


def visualize_segments(clusters: List[Tuple[Segment, Tuple]], image_path, ax=plt):
    """Visualize segment clusters (with assigned color to each segment)"""
    img = None
    if len(clusters):
        segments, color = clusters[0]
        img = draw_lines(mpimg.imread(image_path), segments, color=color)
    for cluster in clusters[1:]:
        segments, color = cluster
        img = draw_lines(img, segments, color=color)
    if img is not None:
        ax.imshow(img)
    return img


palette = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
]  # a colors palette (to use with rotating index)


def visualize_clusters_merge(clusters_dict, segments_cache, image, palette=palette):
    """A helper to visualize/render segments merging result"""
    # Extract and sort merged segments
    merged_segments = sorted(
        filter(
            lambda segment: segment is not None,
            [segments_cache.get(key, None) for key in clusters_dict],
        ),
        key=lambda segment: segment.length,
        reverse=True,
    )
    visualize_segments(
        [
            ([segment], palette[index % len(palette)])
            for index, segment in enumerate(merged_segments)
        ],
        image,
    )


def draw_lane(image, extracted_lane: Dict = {}, output_path: Optional[str] = None):
    """render extracted lane"""
    # TODO: refactor separate concern moving out the saving to a file
    output_image = image.copy()
    if "right" in extracted_lane:
        output_image = draw_lines(
            output_image, [extracted_lane["right"]], color=(0, 255, 0)
        )  # right side in green
    if "left" in extracted_lane:
        output_image = draw_lines(
            output_image, [extracted_lane["left"]], color=(255, 0, 0)
        )  # left in red

    save_status = None
    if output_path:
        save_status = plt.imsave(
            output_path, output_image
        )  # TODO: use cv2.imsave instead
    return output_image
