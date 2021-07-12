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
segments = list(filter(lambda segment: not segment.vertical and not segment.nearly_horizontal(config.get("slope_threshold", 0.5)),
[Segment(item) for item in np.squeeze(HoughLines()(edges_image))]))
```

![alt text][hough_lines_segments]

We then cluster the detected Segments using an agglomerative clustering approach where we iteratively merge the most similar segments into a potentially bigger one. The merging is controlled with a global similarity threshold below which the merge is not allowed. Multiple agglomerative clustering could be combined while shrinking the similarity threshold within a range.

The agglomeration could be exploited in different ways: one approach keeps merging until we have two segments (which are more likely to be the lane sides). Another one more conservative approach is to allow more multiple segments/lines (more than 2) then use either the segment length, the slope, or both to define the two lines representing the lane sides.

The two elected sides of the lane will get a right/left label accordingly to their slopes. Besides, lane sides are extended to their mutual intersection and the intersection with the bottom of the image. The result for the example image is shown below.

![detected lane][detected_lane]

## Segment features/similarity/merging

The segment class implements similarity measures such as `cosine similarity` or exponential kernel on a distance. Moreover, it defines the merging strategy as one of the following:
1 - (among the similar) Extending the longer segments by projecting the ends of the small ones.
2 - Mutually extend each of the similar segments by projecting the ends of the other matched one, then chose the longest merge.
3 - We mutually extend both similar segments; the ends of the extensions are then matched. The result ends are chosen on the segments connecting the extended ones given more weight for the longest initial segment (see details below (Weighted merging examples)).

### Extending a segment with a projection of a point

``` python
from lane_detection import Point, Segment

ss = segments_sample[4]
ref = Point(400, 200) # a test point to project on the segment `ss`
projected = ss.project(ref)
projection = Segment((ref.x, ref.y, projected.x, projected.y))

extension = Segment((ss.x1, ss.y1, projected.x, projected.y))
extension_image = visualize_segments([([ss], [255, 0, 0]), # original segment in red
                    ([projection], [255, 255, 0]), # ref/test point ortho. projection on the red segment
                    ([extension], [0, 0, 0]), # resulted extension
                  ], test_images[0])

```

![Segment extension with point projection][segment_extension]

### Weighted merging examples
```python
synth_segments = [Segment((400, 300, 200, 500)), Segment((500, 300, 100, 500))] # test synthetic segments
extended = synth_segments[0].mutual_extend(synth_segments[1], sort=True, reverse=True)

# Match extended segment ends
matching = extended[0].match_ends(extended[1])

match_1 = Segment(extended[0].ends[0].as_tuple + extended[1].ends[matching[0]].as_tuple)
match_2 = Segment(extended[0].ends[1].as_tuple + extended[1].ends[matching[1]].as_tuple)


weight = extended[1].norm/( extended[1].norm + extended[0].norm )


t1_x, t1_y = (match_1.vect*weight).ravel()
t2_x, t2_y = (match_2.vect*weight).ravel()


end_0_translated = Segment(extended[0].ends[0].as_tuple + extended[0].ends[0].translate(Point(t1_x, t1_y)).as_tuple)
end_1_translated = Segment(extended[0].ends[1].as_tuple + extended[0].ends[1].translate(Point(t2_x, t2_y)).as_tuple)

merge_output = Segment(end_0_translated.ends[1].as_tuple + end_1_translated.ends[1].as_tuple)

weighted_merge_image = visualize_segments([ (extended, [0, 0, 0]), # extended Segments in black
                     (synth_segments, [255, 0, 0]), # original segments in RED
                     ([match_1, match_2], [255, 255, 0]), # matched ends (in yellow)
                     ([end_0_translated, end_1_translated], (0, 0, 255)), # where the merge will happen
                     ([merge_output], (0, 255, 0)) # merge output in green
                   ],test_images[0] )
```

![Weighted merge][weighted_merge]


### Drawing the lanes

The `draw_lanes()` function consumes the output of the lane extraction: a dictionary of labeled segments (keys are the labels) as right and left according to their slope and the `Segment` as a  value (see example below).


```JSON
{"left": "(467, 302) -> (892, 546)",
 "right": "(280, 460) -> (467, 303)"}
```

```python
def draw_lane(image, extracted_lane:Dict={}, output_path:Optional[str]=None) :
    """ render extracted lane """
    output_image = image.copy()
    if "right" in extracted_lane:
        output_image = draw_lines(output_image, [extracted_lane["right"]], color=(0, 255, 0)) # right side in green
    if "left" in extracted_lane:
        output_image = draw_lines(output_image,  [extracted_lane["left"]], color=(255, 0, 0)) # left in red

    save_status = None
    if output_path:
        save_status = plt.imsave(output_path, output_image) # TODO: use cv2.imsave instead
    return output_image
```

## Potential shortcomings with the current pipeline

One potential shortcoming is that the approach relies to good and clean line detection.  A noisy hough output will result in a challenging clustering problem.
As the challenging video case presents the shadow of a tree on the road could cause a challenging segments pool to merge, with potentially more risk of extending a false positive segment. This also might occur with a shadow of tree trunk or a street light.

Another shortcoming could be the calibration of the parameters for the similarity a too high threshold will not allow a merging of the segments while a too permissive one would result merging/extending a slightly off direction that does not fit the lane.

As each of lane side has on its turn two sides (would result in 2 ). Oscillating side definition between the 2 edges of a side (the longer is extended)


## Possible improvements

### Band context lane detection

A possible improvement is splitting the image vertically and then performing local segments/lines/lane detection within each band. Do a merging within each of the bands, then perform an inter bands merging the detected segments.

### Injecting prior from the previous frame

Instead of no prior approach where the lane detection has only the current frame/image as input, we can think about transferring information from the previously processed frame/image. That can be ensured in various ways:
1- A narrower filtering around the previous slopes ranges.
2- Split the previous lane and inject the near context (bottom of the image/ closer to the car)
3 - Extend the further segments from the last frame and inject them into the pool of segments.