import re
from collections import defaultdict

from setuptools import find_packages, setup


# Gross way to have the version only in one place ...
def get_version():
    version_pattern = re.compile(r'^__version__ = "(?P<version>[0-9]+\.[0-9]+\.[0-9]+)"\n$')
    with open("awareutils/__init__.py") as f:
        for line in f:
            m = version_pattern.match(line)
            if m:
                return m.group("version")
        raise RuntimeError("Failed to find version!")


def get_extras_require():
    with open("extras-requirements.txt") as f:
        extra_deps = defaultdict(set)
        for line in f:
            line = line.strip()
            if ":" in line:
                module, tagnames = line.split(":")
                for tag in tagnames.split(","):
                    extra_deps[tag.strip()].add(module)

        # add all, and add all to test (as we assume the tests need all packages to run)
        all_non_test = set(module for tag, modules in extra_deps.items() for module in modules if tag != "test")
        extra_deps["all"] = all_non_test
        for module in all_non_test:
            extra_deps["test"].add(module)
    print(extra_deps)
    return extra_deps


setup(
    name="awareutils",
    packages=find_packages(),
    version=get_version(),  # NOQA
    install_requires=["numpy>=1.19", "loguru>=0.5.3"],
    extras_require=get_extras_require(),
)
