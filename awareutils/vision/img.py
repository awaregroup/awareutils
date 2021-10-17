import json
from enum import Enum, auto, unique
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from loguru import logger

# Import only what we need
try:
    import cv2
except ImportError:
    from awareutils.vision.mock import cv2
try:
    import PIL.Image as PILImageModule
except ImportError:
    from awareutils.vision.mock import PILImageModule
try:
    import piexif
except ImportError:
    from awareutils.vision.mock import piexif


@unique
class ImgType(Enum):
    BGR = auto()
    RGB = auto()
    PIL = auto()


class ImgSize:
    def __init__(self, *, h: int, w: int):
        self._validate_img_shape(w=w, h=h)
        self.w = w
        self.h = h

    def __eq__(self, other):
        return self.w == other.w and self.h == other.h

    def __hash__(self):
        return hash((self.w, self.h))

    @staticmethod
    def _validate_img_shape(*, w: int, h: int):
        if not isinstance(w, int) or not isinstance(h, int):
            raise ValueError("w and h should be ints")
        if w <= 1 or h <= 1:
            raise ValueError("img should be at least 1x1")


# Uggh, circular imports. Other things e.g. shape want ImgSize, so let's give them that first above. Note that we could
# define ImgSize in a different file which would also resolve the circular imports, but I reckon it's pretty tied to
# Img, so let's leave it here and accept this slight hack.
from awareutils.vision.shape import Rectangle


class Img:
    """
    Our image class that's:
        - Optimize to do no conversions if you have to i.e. if you always work with rgb this will be just as fast as
            using rgb arrays, and likewise bgr or PIL. Nice utilities for converting between formats.
        - Nice attributes like height/width/etc.
        - Custom json-serializable metadata store in the UserComment field of EXIF.
    """

    def __init__(
        self,
        source: Union[np.ndarray, "PILImageModule.Image"],
        itype: ImgType,
        metadata: Optional[Dict] = None,
        make_arrays_contiguous: bool = True,
    ):
        self.source = source
        self.itype = itype
        self._metadata = metadata
        self.make_arrays_contiguous = make_arrays_contiguous

        if self.itype == ImgType.PIL:
            if not isinstance(source, PILImageModule.Image):
                raise RuntimeError("Set source type to PIL but source isn't a PIL Image")
        elif self.itype in (ImgType.RGB, ImgType.BGR):
            if not isinstance(source, np.ndarray):
                raise RuntimeError(f"Set source to {self.itype.name} but isn't an np.ndarray")
            if not source.flags.c_contiguous:
                if make_arrays_contiguous:
                    source = np.ascontiguousarray(source)
                else:
                    logger.warn("Source isn't contiguous which can cause subtle OpenCV problems.")

        else:
            raise RuntimeError("Unknown source type")

        self._draw = None

        # Set some attributes for performance (as they are commonly accesses and often in loops etc.):
        self.isize = self._get_size()
        self.h = self.isize.h
        self.w = self.isize.w

    def _get_size(self) -> ImgSize:
        if self.itype == ImgType.PIL:
            h, w = self.source.height, self.source.width
        elif self.itype in (ImgType.BGR, ImgType.RGB):
            h, w = self.source.shape[:2]
        return ImgSize(w=w, h=h)

    @property
    def metadata(self) -> Optional[Dict]:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict) -> None:
        if not isinstance(value, Dict):
            raise ValueError("value must be a dict")
        self._metadata = value

    def rgb(self) -> np.ndarray:
        if self.itype == ImgType.PIL:
            arr = np.array(self.source)
        elif self.itype == ImgType.RGB:
            arr = self.source
        elif self.itype == ImgType.BGR:
            arr = cv2.cvtColor(self.source, cv2.COLOR_BGR2RGB)

        # Make contiguous if required:
        if self.make_arrays_contiguous:
            return np.ascontiguousarray(arr)
        return arr

    def bgr(self) -> np.ndarray:
        if self.itype == ImgType.PIL:
            arr = cv2.cvtColor(np.array(self.source), cv2.COLOR_RGB2BGR)
        elif self.itype == ImgType.RGB:
            arr = cv2.cvtColor(self.source, cv2.COLOR_RGB2BGR)
        elif self.itype == ImgType.BGR:
            arr = self.source

        # Make contiguous if required:
        if self.make_arrays_contiguous:
            return np.ascontiguousarray(arr)
        return arr

    def pil(self) -> "PILImageModule.Image":
        if self.itype == ImgType.PIL:
            return self.source
        elif self.itype == ImgType.RGB:
            return PILImageModule.fromarray(np.ascontiguousarray(self.source))
        elif self.itype == ImgType.BGR:
            return PILImageModule.fromarray(np.ascontiguousarray(self.bgr()))

    @classmethod
    def open(cls, path: Union[Path, str], itype: ImgType, load_metadata: bool = False) -> "Img":
        if not isinstance(path, (Path, str)):
            raise ValueError("path must be a Path or str")
        path = str(path)
        if not load_metadata:
            if itype == ImgType.BGR:
                source = cv2.imread(path)
                if source is None:
                    raise RuntimeError("Failed to read with cv2")
                return cls(source=source, itype=ImgType.BGR)
            elif itype == ImgType.RGB:
                source = cv2.imread(path)
                if source is None:
                    raise RuntimeError("Failed to read with cv2")
                source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
                return cls(source=source, itype=ImgType.RGB)
            elif itype == ImgType.PIL:
                return cls(source=PILImageModule.open(path), itype=ImgType.PIL)
        else:
            pil = PILImageModule.open(path)
            meta = None
            if load_metadata:
                meta = pil.info.get("exif")
                if meta is not None:
                    meta = piexif.load(meta)["Exif"][piexif.ExifIFD.UserComment]
                    meta = json.loads(meta)
            if itype == ImgType.BGR:
                rgb = np.array(pil)
                return cls(source=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), itype=ImgType.BGR, metadata=meta)
            elif itype == ImgType.RGB:
                return cls(source=np.array(pil), itype=ImgType.RGB, metadata=meta)
            elif itype == ImgType.PIL:
                return cls(source=pil, itype=ImgType.PIL, metadata=meta)

    @classmethod
    def open_pil(cls, path: Union[Path, str], load_metadata: bool = False) -> "Img":
        return cls.open(path=path, itype=ImgType.PIL, load_metadata=load_metadata)

    @classmethod
    def open_bgr(cls, path: Union[Path, str], load_metadata: bool = False) -> "Img":
        return cls.open(path=path, itype=ImgType.BGR, load_metadata=load_metadata)

    @classmethod
    def open_rgb(cls, path: Union[Path, str], load_metadata: bool = False) -> "Img":
        return cls.open(path=path, itype=ImgType.RGB, load_metadata=load_metadata)

    def save(self, path: Union[Path, str], save_metadata: bool = True, **kwargs) -> None:
        if not isinstance(path, (Path, str)):
            raise ValueError("path must be a Path or str")
        path = str(path)
        should_save_metadata = save_metadata and self._metadata is not None
        # If we're saving metadata, it's got to be PIL:
        if self.itype == ImgType.PIL or should_save_metadata:
            pil = self.pil()
            if should_save_metadata:
                if "exif" in kwargs:
                    raise RuntimeError(
                        "We're saving metadata in EXIF already, so it's unsupported for you to use it too!"
                    )
                exif_dict = {"Exif": {piexif.ExifIFD.UserComment: json.dumps(self._metadata).encode("utf8")}}
                kwargs["exif"] = piexif.dump(exif_dict)

            # Default to optimize=True. This means we'll get failures for any save methods that don't supported
            # `optimize` but since these are uncommon (and the user can fix this by setting optimize=False), meh.
            if "optimize" not in kwargs:
                kwargs["optimize"] = True

            pil.save(path, **kwargs)
        elif self.itype == ImgType.RGB:
            cv2.imwrite(path, cv2.cvtColor(self.source, cv2.COLOR_RGB2BGR))
        elif self.itype == ImgType.BGR:
            cv2.imwrite(path, self.source)

    def resize(self, isize: ImgSize) -> "Img":
        """
        Resize the image
        """
        # TODO: support different sampling
        # TODO: test
        if self.itype == ImgType.PIL:
            return Img(source=self.source.resize(size=(isize.w, isize.h)), itype=self.itype, metadata=self._metadata)
        elif self.itype in (ImgType.RGB, ImgType.BGR):
            return Img(source=cv2.resize(self.source, (isize.w, isize.h)), itype=self.itype, metadata=self._metadata)

    def crop(self, rectangle: Rectangle, copy: bool = False) -> "Img":
        if rectangle._isize != self.isize:
            raise RuntimeError(
                (
                    "This crop rectangle has a different size to the img i.e. the coordinate systems don't match. "
                    "Consider using rectangle.project() first."
                )
            )
        if self.itype == ImgType.PIL:
            if not copy:
                raise RuntimeError("PIL crops are always copys. I think?")
            return Img(
                source=self.source.crop((rectangle.x0, rectangle.y0, rectangle.x1 + 1, rectangle.y1 + 1)),
                itype=self.itype,
                metadata=self._metadata,
            )
        elif self.itype in (ImgType.RGB, ImgType.BGR):
            source = rectangle.slice_array(self.source)
            if copy:
                source = source.copy()
            return Img(source=source, itype=self.itype, metadata=self._metadata)

    @classmethod
    def from_bgr(cls, array: np.ndarray, metadata: Optional[Dict] = None, make_arrays_contiguous: bool = True):
        return cls(source=array, itype=ImgType.BGR, metadata=metadata, make_arrays_contiguous=make_arrays_contiguous)

    @classmethod
    def from_rgb(cls, array: np.ndarray, metadata: Optional[Dict] = None, make_arrays_contiguous: bool = True):
        return cls(source=array, itype=ImgType.RGB, metadata=metadata, make_arrays_contiguous=make_arrays_contiguous)

    @classmethod
    def from_pil(cls, pil: PILImageModule.Image, metadata: Optional[Dict] = None):
        return cls(source=pil, itype=ImgType.PIL, metadata=metadata)

    @classmethod
    def new(
        cls,
        size: ImgSize,
        itype: ImgType,
        metadata: Optional[Dict] = None,
    ) -> "Img":
        if itype == ImgType.PIL:
            return Img(
                source=PILImageModule.new("RGB", size=(size.w, size.h), color=(0, 0, 0)), itype=itype, metadata=metadata
            )
        elif itype in (ImgType.RGB, ImgType.BGR):
            return Img(source=np.zeros((size.h, size.w, 3), np.uint8), itype=itype, metadata=metadata)

    @classmethod
    def new_pil(
        cls,
        size: ImgSize,
        metadata: Optional[Dict] = None,
    ) -> "Img":
        return cls.new(size=size, itype=ImgType.PIL, metadata=metadata)

    @classmethod
    def new_bgr(
        cls,
        size: ImgSize,
        metadata: Optional[Dict] = None,
    ) -> "Img":
        return cls.new(size=size, itype=ImgType.BGR, metadata=metadata)

    @classmethod
    def new_rgb(
        cls,
        size: ImgSize,
        metadata: Optional[Dict] = None,
    ) -> "Img":
        return cls.new(size=size, itype=ImgType.RGB, metadata=metadata)


# Ewwww, getting around circular imports ...
from awareutils.vision.draw import Drawer, OpenCVDrawer, PILDrawer


@property
def draw(self) -> Union[Drawer]:
    if self._draw is None:
        if self.itype == ImgType.PIL:
            # TODO: don't hardcode last arg
            self._draw = PILDrawer(img=self, reproject_shapes_if_required=True)
        elif self.itype in (ImgType.RGB, ImgType.BGR):
            self._draw = OpenCVDrawer(img=self, reproject_shapes_if_required=True)
    return self._draw


Img.draw = draw
