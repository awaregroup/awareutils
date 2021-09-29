import re

from setuptools import find_packages, setup

# Gross way to have the version only in one place ...
version_pattern = re.compile(r'^__version__ = "(?P<version>[0-9]+\.[0-9]+\.[0-9]+)"\n$')
with open("awareutils/__init__.py") as f:
    for line in f:
        m = version_pattern.match(line)
        if m:
            version = m.group("version")
            break
    else:
        raise RuntimeError("Failed to find version!")

setup(
    name="awareutils",
    packages=find_packages(),
    version=version,  # NOQA
    install_requires=["piexif>=1.1.3", "pillow>=7.2.0", "numpy>=1.19"],
)
