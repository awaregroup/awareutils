# Decisions in awareutils

## Explicit is best

We force keyword arguments for most things, as it's too easy to get `x` and `y` (or `width` and `height`) the wrong way round, especially given `OpenCV` is a bit confusing in this regard. It's maybe a little tedious if you know what you're doing, but it reduces bugs and makes your code more readable for others.

## Coordinates are complicated ...

Think about the box `x0, y0, x1, y1 = 0, 0, 1, 1` - is that a `1x1` or a `2x2` box? Well, it's both, which can be messy. If you do `img[y0:y1, x0:x1, :] = 1` it's only a `1x1` box. But if you do `cv2.rectangle(img, (x0, y0), (x1, y1), (1, 1, 1))` then it's a `2x2` box. If this were only drawing boxes, no major - but it can mean messy calculations etc. when you're dealing at the maximum of the dimension. For example, if your image is `1920` wide, is `x1 = 1920` valid, or should `x1 = 1919` be the last value? As above, either can be correct.

We should consider the `x` and `y` not as positions, but simply the index of the pixel. With this in mind, `x1 = 1920` is invalid, as it implies there are `1921` elements (`0`-based indexing). Thus, if you want a full box you'd have `x0 = 0, x1 = 1919` and hence if you want just a single pixel box, you'd have `x0 = 0, x1 = 0`. This matches `cv2.rectangle` behaviour. The only downside is that, as with normal Python, `numpy` array indexing `img[:, x0:x1, :]` would be flawed as that only includes up to `x1 - 1`, which then gets really confusing. Also, simple things like width become the slightly more confusing `x1 - x0 + 1`. We could hence just adopt the Python/numpy convention of `x1 = 1920` and adapt other functions (like `cv2.rectangle` to know we mean "up to `x1 - 1` but not including `x1`"). The downside of this is that we're dealing with more than rectangles - we've got points (in which `x1 != x0` seems a bit weird), or polygons, etc., where you're not drawing in simple `x0:x1` limits, and it makes more sense that all points specifying the shape are included. In addition, it's really easy to fall into the `img[bbox.y1, bbox.x1, :]` trap (or similar) which is invalid at e.g. `x1 = 1920`.

Finally, it's worth considering normalized coordinates. These are really useful as they're the 'same' even under image resizing, which is really handy. However, they're a bit tricky, as this is using a continuous space to represent a discrete one. For example:

- Should they be `[0, 1]` or `[0, 1)` i.e. should `1` ever be valid? If it's not, that gets really confusing - how do you specify the right-hand side of a box? `x0 = 0.9999999...`? This ultimately comes down to us choosing th
- What's a point? In normalized coordinates, a point (i.e. infinitely small point in space) is fine. However, think of a camera or an image in pixels - a single pixel point is actually a box defined by the left/right/top/bottom of that pixel. So this gets a bit messy.
- Normalized coordinates imply higher accuracy than we actually have, and we'll get weird rounding errors. E.g. if you have a `1000x1000` image, and points at `x = 0` and `x = 1` these are `x = 0.0` and `x = 0.001` respectively, in normalized coordiantes. If we then apply them to a `100x100` image, they both become `x = 0` in pixel coordinates. This is intended behaviour, but the fact that two distinct bounding boxes become the same is potentially non-trivial and error causing.

So, what do we do?

- Recognize that a coordinate system is inherently tied to the image size - think like the pixels in an image sensor. That is `x = 1` only makes sense for that particular image size. So - let's be explicit about that.
- On that note, let's avoid normalized coordinates completelyj:w
, as ultimately, pretending a discrete coordinate systems is (bounded) continuous is too messy, and this also saves us having to check between normalized and pixel coordiantes. It's sad, as they are convenient and we'll have to do something else to protect against misapplying pixel coordinates from one system (image size) to another ... though we have to do this anyway (i.e. protecting against normalized vs not).
- Let's stick with OpenCV's approach such that all points that define an object should be contained in the image i.e. `x1 = 1920` is invalid. This is mostly as it's easier to think about, matches the behaviour of non-rectangular shapes, and reduces the `img[y1, x1, :]` bug.
- We're not going to use the term "point" as that implies and infinitesimal thing, whereas we actually deal with pixels which do have size. Let's use "pixel" instead i.e. you're refering to a specific pixel, not a point.
- To get around the Python/`numpy` indexing issue (i.e. users needing to remember `img[:, x0:x1 + 1, :]`), let's create some utility functions - see below.

## Slicing shapes

It's pretty common to do things like `bgr[b.y0:b.y1, b.x0:b.x1, :]` but, as above, that can be tricky with indexing. So we could just create our own `slice()` method which returns the slices (including using python `slice` which is really nice). However, it doesn't really make sense for anything except (unrotated) rectangles i.e. which can be defined by an `x` and `y` range. For things like circles or polygons etc. - no such luck. So what we really want is something like `shape.select_pixels(img)` and then to e.g. set sum the image pixels where the shape is you'd do `shape.select_pixels(img).sum()`. For polygons etc., this is likely going to just be using the shape mask (i.e. `bgr[shape.mask(img)].sum()`) - and all we really do with polygons is short-circuit this for performance. (Using index vs a mask is orders of magntiude faster for e.g. assigning to an array.)

However, indexing numpy arrays with masks [is weird](https://numpy.org/doc/stable/user/basics.indexing.html#assigning-values-to-indexed-arrays). Specifically, if you just index the array to get the values, you get a *copy* of the original array - but if you assign while indexing, it mutates the original array. That is

```python
>>> x = np.arange(10) 
>>> mask = x > 5
>>> x[mask] = 10
>>> x
array([ 0,  1,  2,  3,  4,  5, 10, 10, 10, 10])
```

Does what you'd expect. This doesn't.

```python
>>> x = np.arange(10) 
>>> mask = x > 5 
>>> sliced_arr = x[mask]
>>> sliced_arr[:] = 10 
>>> sliced_arr
array([10, 10, 10, 10])
>>> x
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

Why? Because `sliced_arr` is a copy, so assigning to it doesn't change `x`. (Also remembering the `[:]` is a bit of a trick for new players.) Point is our above `shape.select_pixels(img)` won't work at all for assigning values, as at the end of our function we'd do something like `return bgr[shape.mask()]`. Since that creates a copy, we could never do nice things like changing colors etc. Also, as a side note, slicing with indices preserves dimensions, whereas slicing with a mask doesn't i.e.

```python
>>> x = np.ones((1920, 1080, 3), np.uint8) 
>>> mask = np.ones((1920, 1080), bool)
>>> mask[10:100, 10:100] = True
>>> x[mask].shape
(2073600, 3)
>>> x[10:100, 10:100, :].shape
(90, 90, 3)
```

A recipe for confusion.

So, what do we do? Well:

- Slicing in the sense of python `slices` only really makes sense for rectangles i.e. `bgr[b.y0:b.y1, b.x0:b.x1, :]`. So let's just add a custom `slice_array` method for that. For now, let's have `rectangle.slice_array(img)` return the sliced array directly, as that way we can guarantee that `img` is of the same pixel coordinates as the shape. However, since what we're really doing in the above case is just cropping the image, let's provide a method for that i.e. `img.crop(rectangle)` as that's a lot more obvious.
- 'Slicing' with masks is risky, so let's not do it and, for now, leave it to the user. As above, much of the time we'll aim to provide convenience method (e.g. `shape.fill(bgr, 255)` to set it to `255` under the shape) so masking won't be used. But let's still provide a `shape.mask()` method so users can do what they wish with the mask themselves. (Masks are a lot more sensible with things like `np.logical_and` etc., or OpenCV functions.)

### OpenCV and PIL?

Yup, 'cos some libraries pick one or the other, so it's nice to be able to use one or the other under the hood. For now, let's see how it goes, and if it's too painful, we'd probably go with just OpenCV. 

Assuming we're going with both, how do we allow people to pick just one without having them both imported? Well, see `awareutils/vision/mock.py` and it's use in `awareutils/vision/img.py` - basically, if we can't import OpenCV or PIL, we continue as fine until it's attempted to be used, in which case the `Mock`s `__getattr__` fires an exception.

### Rectangles are defined by two pixels

Yup - definition by pixels makes it easier and more consistent with other shapes. We've got `from_x0y0x1y1` and `from_xywh` for when you want to shortcut.

### Words and spelling

We chose the following spellings and encourage their use as a) it adds consistency (by avoiding having various different spellings), b) they're brief and easy to type, and c) it keeps us distinct from other libraries (which chose one of the correct spellings).

- `img` (vs `img|image`)
- `col` (vs `col|color|colour`)
- `fidx` to refer to the frame index (as opposed to `fidx|idx|i|index` etc.)
