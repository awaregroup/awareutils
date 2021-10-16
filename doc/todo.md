# TODO

- tests!
- This is ugly `line = Line(isize=img.size, p0=Pixel(isize=img.size, x=10, y=10), p1=Pixel(isize=img.size, x=900, y=900))`. Could allow first argument without keyword, i.e. `Line(img.size, p0=...)` which makes it shorter. Maybe try `line = img.line(p0=img.Pixel(x=10, y=10), p1=img.Pixel(x=900, y=900))` though it's still a bit tedious. What we really want is 
  ```python
  with img.size:
    line = Line(p0=Pixel(x=10, y=10), p1=Pixel(x=900, y=900))
  ```
  But how to achieve that? Would need weird introspection at best. Maybe we just do `img.Line` and `img.Pixel` ...
- push to pypi
- Add a pi camera class
- Should a rect defined by two points? It'd be more consistent with the rest of shape stuff where lines etc. are defined in terms of points.
- Admin:
  - flake8 config and generally nicer build stuff.
  - build pipeline in github.
- Performance:
  - Cythonize some of the shape stuff (coordinate checking etc.) to keep that fast.
  - Memoise calls like shape.center or shape.mask so if it's called a second time it's cached.
- Features:
  - Drawing:
    - Transparency?
    - Add shape drawing methods
      - arrow? maybe a few other shapes.
      - nicer text! add some cool fonts?
  - OpenCV:
    - Automatically search USB devices if you don't know which number (as it's a bit weird) ...
    - Automatically search supported fourcc in videowriter (or does -1 do that?). At least find some good fourcc defaults.
  - Add fast video seeking to FileCapture.
  - Shape:
    - `shape.bounding_box` method
    - `shape.mask` method
  - Add v4l2 loopback videowriter
  - Using a 2d mask on a 3d image. Or maybe just have a `shape.mask3d()` method.
- Figure out nicer solution for circular imports