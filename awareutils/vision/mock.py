class Mock:
    def __init__(self, name: str):
        self.__name = name

    def __getattr__(self, name: str):
        raise RuntimeError(
            (
                f"Module {self.__name} is needed for this functionality. Please consider installing it as an "
                "extras_requires i.e. `pip install awareutils[<appropriate tag>]`. See extras-requirements.txt or the "
                "documentation."
            )
        )


cv2 = Mock("cv2")
PILImageModule = Mock("PIL")
piexif = Mock("piexif")
