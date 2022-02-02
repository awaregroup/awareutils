# awareutils

A collection of Python utilities used within Aware Group. Generally computer vision for now. The objective is:

- Reduce common errors or sources of confusion (see below).
- Improve development time by a) getting features for free (e.g. threaded cameras) and b) getting 'debug' features for free (like printing FPS) which will (hopefully) help when issues are encountered.
- Improve quality (performance etc.).
- Standardize code across various projects.
- Being opinionated and explicit. This helps deliver the above objectives, and speed up learning.

## Installation

```sh
git clone https://github.com/awaregroup/awareutils
pip install awareutils[all]
```

The above installs everything including e.g. `OpenCV` and `Pillow`. If you know you're only going to work with `OpenCV` then this library supports that - in this case you'd `pip install awareutils[cv2]`. See `extras-requirements.txt` for a breakdown.

## Features

- Single `Img` class. The key features are:
  - Clarity between BGR / RGB / PIL, and no performance cost for using a particular one. If you only ever want `PIL` you don't need to install `OpenCV`.
  - Ability to save metadata (e.g. annotations) in the image itself (via EXIF metadata).
- Various video classes which:
  - Are threaded for optimal performance (i.e. your main loop isn't hanging on decoding).
  - Can (hopefully) be easily extended for other cameras - just define how to open/close the camera, and read a frame.
  - OpenCV specific classes for:
    - Reading from a file - you can choose to read every frame sequentially, or simulate it as a live camera with a given FPS.
    - Reading from a live (USB) source, including nicely setting attributes like height etc.
    - Writing to a file. Eventually there'll be a nice way to get the 'right' fourcc without all the messing round.
- One standard approach to coordinate systems. Note OpenCV vs numpy indexing or normalised vs not coordinates, etc.
- Shapes ...

You can see more about some of the design decisions in [./doc/decisions.md](./doc/decisions.md).

### Bugs we're trying to cover

- General bugs by rolling your own = ) Use this simply 'cos it's been used before, and is tested.
- Usually `cv2.rect(...)` etc. edit the image in place ... except when it's not contiguous. Hence we use `np.ascontiguousarray` etc. when we're looking at arrays.
- Confusiong around RGB vs BGR. So we're explicit.
- Confusion between PIL and OpenCV. Support both, and conversion between.
- Separation of annotations from image - so we allow saving metadata in the image EXIF metadata.
- When reading an OpenCV camera, the buffer can lag - so do it in a thread. Likewise, thread it all up for performance.
- Pixel coordinates (vs normalised) are confusing. See above. Fix by being a little explicit and only using pixel coordinates, not using the term "Point", etc.
- Invalid shape coordinates (e.g. outside image etc.). Check for this, and fix, etc.
- Bounding box formats (x0y0x1y1, x0y0wh, xcycwh, etc.) - be explicit. Likewise `(w, h)` vs `(h, w)` - be explicit by forcing kwargs.
- Annoyance setting up OpenCV windows fullscreen etc. Provide some utils.
- Threadig. It's the way to do things faster, but it's hard to do it right with the right features.

## Developing

```sh
pip install -e .
```

Then you can edit at will, and ensure you run

```sh
pytest .
```

If you want to see some logs, ensure you do `logger.enable('awareutils')`.
