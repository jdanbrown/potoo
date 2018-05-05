try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements
from setuptools import find_packages, setup

setup(
    name='potoo',
    packages=find_packages(exclude=[]),
    # Parse from reqs.txt
    #   - https://stackoverflow.com/a/16624700/397334
    install_requires=[str(x.req) for x in parse_requirements('requirements.txt', session='dummy')],
)
