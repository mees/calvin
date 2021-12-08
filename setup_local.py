#!/usr/bin/env python

"""Setup Calvin installation."""

from os import path as op
import re

from setuptools import find_packages, setup


def _read(f):
    return open(op.join(op.dirname(__file__), f)).read() if op.exists(f) else ""


_meta = _read("calvin_models/calvin_agent/__init__.py")


def find_meta(_meta, string):
    l_match = re.search(r"^" + string + r'\s*=\s*"(.*)"', _meta, re.M)
    if l_match:
        return l_match.group(1)
    raise RuntimeError(f"Unable to find {string} string.")


meta = dict(
    name=find_meta(_meta, "__project__"),
    version=find_meta(_meta, "__version__"),
    license=find_meta(_meta, "__license__"),
    description="CALVIN - A benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks",
    platforms=("Any"),
    zip_safe=False,
    keywords="pytorch Lfp".split(),
    author=find_meta(_meta, "__author__"),
    author_email=find_meta(_meta, "__email__"),
    url=" https://github.com/mees/calvin",
    packages=find_packages(exclude=["tests"]),
)

if __name__ == "__main__":
    print("find_package", find_packages(exclude=["tests"]))
    setup(**meta)
