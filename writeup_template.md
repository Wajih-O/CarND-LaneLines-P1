# **Finding Lane Lines on the Road**


**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[//]: # (Image References)

[image1]: ./test_images/solidWhiteCurve.jpg "original"

[interest_edges]: ./examples/interest_edges.jpg "Edges within interest region"

[hough_lines_segments]: ./examples/detected_segments.jpg "Detected lines/segments"

[detected_lane]: ./examples/detected_lane.jpg " Detected lane "

[segment_extension]: ./examples/segment_extension.jpg "Segment extension with a projected point"

[weighted_merge]: ./examples/weighted_merge.jpg "weighted merge"
---

## Pipeline Outline


The proposed pipeline consisted of several steps implemented in `lane_detection.utils.extract_lane` function.

First, the image (or frame) is converted to gray-scale then a gaussian blur filter is applied as part of edge detection performed with a canny filter. The edges/contour image is then cropped to a region of interest (and optionally here, the area above the horizon priorly estimated using `horizon_crop` helper)

![alt text][image1]

``` python
edges_image = interest_region_crop(horizon_crop(canny(gaussian_blur(grayscale(image)))))
```

![alt text][interest_edges]

The output is then an edge image containing the piece of the road, including the lane lines. On this resulted image, we perform a line detection using HoughLines see `lane_detection.hough_lines`. These lines are filtered to remove the lines with extreme slopes:
horizontal,  nearly horizontal, and vertical lines. As an output of these lines, detection and filtering a list of `Segment` is built. The `Segment` class (see `lane_detection/segment.py`) is a crucial helper class that implements several geometrical helpers to extract slope, similarities, projection, and intersection.


```python

    # Extract segments (detect lines) then filter the nearly horizontal and nearly vertical ones :)
    hough = HoughLines(min_line_length=20)
    segments = list(
        filter(
            lambda segment: not segment.nearly_vertical(0.2)
            and not segment.nearly_horizontal(config.get("slope_threshold", 0.8)),
            [Segment(item) for item in np.squeeze(hough(edges_image))],
        )
    )

```

![alt text][hough_lines_segments]

If a previous detection is available, we this as a prior (see code snippet below)

``` python
    # Injecting prior
    if prior is not None:
        # only get the lower part of segment with proportion prop (the proportion is valid because of a weighted poly/line-fit using the segment length)
        segments.extend(map(lambda segment: segment.lower(prop=.2), prior.values()))
```

To separate the slopes into two clusters, one for each side of the lane (right/left), we first fit a line on the completed data that most likely
generates a line that separates the two clusters. Using the slope of this fitted line as a threshold, we split the segments into our left and right groups/classes.

``` python
    # Split slope extraction (use a fitted line to split the lane into right side and left side)
    X, y = zip(*[(end.x, end.y) for segment in segments for end in segment.ends.values()])
    split_slope = np.polyfit(X, y, 1)[0]
    # split into 2 classes
    side_1_segments, side_2_segments = [], []
    for segment in segments:
        (side_1_segments if segment.slope < split_slope else side_2_segments).append(segment)
```


Having the two clusters of segments, we then fit a line on each. Then, for each of the two clusters,  adjust one segment to the fitted line; projecting the segment with x coordinate on the fitted line. Preferably the longest segment for lower artifacts (see `Segment.y_adjust` method). The two adjusted segments are then extended to the bottom of the screen and the horizon (defined by their intersection) and returned as a right/left annotated result to the drawing utils.

![detected lane][detected_lane]

Optionally we can merge/clean the extracted segments using an agglomerative clustering approach (see `lane_detection/clustering.py`) where we iteratively merge the most similar segments into a potentially bigger one. The merging is controlled with a global similarity threshold below which the merge is not allowed. Multiple agglomerative clustering could be combined while shrinking the similarity threshold within a range. That would remediate to false detection (no segments) for a high hough line length threshold, using a more permissive threshold and agglomerate the small segment into a bigger one.


### Drawing the lanes

The `draw_lanes()` function consumes the output of the lane extraction: a dictionary of labeled segments (keys are the labels) as right and left according to their slope and the `Segment` as a  value (see example below).


```JSON
{"left": "(467, 302) -> (892, 546)",
 "right": "(280, 460) -> (467, 303)"}
```

```python
def draw_lane(image, extracted_lane: Dict = {}, output_path: Optional[str] = None):
    """render extracted lane"""
    lane_annotation_image = image.copy()
    if "right" in extracted_lane:
        lane_annotation_image = draw_lines(
            lane_annotation_image, [extracted_lane["right"]], color=(0, 255, 0),
        thickness=10)  # right side in green
    if "left" in extracted_lane:
        lane_annotation_image = draw_lines(
            lane_annotation_image, [extracted_lane["left"]], color=(255, 0, 0),
        thickness=10)  # left in red
    output_image = weighted_img(lane_annotation_image, image, .5, .5)
    save_status = None
    if output_path:
        save_status = plt.imsave(
            output_path, output_image
        )
    return output_image
```

## Potential shortcomings with the current pipeline

One potential shortcoming is that the approach relies on clean line detection.  A noisy hough output will result in a challenging clustering/lane sides separation problem.

As the challenging video case presents the shadow of a tree on the road and the imperfection of the road surface due to reparation could cause a hard segments pool to filter/merge, with potentially more risk of extending a false positive segment (if we use agglomerative approach). We should be aware of high/low contrast while extracting the contour/lines.

Another shortcoming is the linear model that could not fit curves, and it is better to partition the image (vertically/depth-wise) and stitch band detected lanes together (see possible improvements).

## Possible improvements

### Band context lane detection

A possible improvement is splitting the image vertically and then performing local segments/lines/lane detection within each band. Do a merging within each of the bands, then perform an inter bands merging the detected segments.

### Injecting prior from the previous frame

Instead of no prior approach where the lane detection has only the current frame/image as input, we demonstrate one wat of prior injection from a previous frame with successful detection. We can think about another form of prior as re-use the original segments or the one with high confidence. (with a configurable prior weight)