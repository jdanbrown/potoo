from pip.req import parse_requirements
from setuptools import setup, find_packages

setup(
    name='potoo',
    packages=find_packages(exclude=[]),
    # Parse from reqs.txt
    #   - https://stackoverflow.com/a/16624700/397334
    install_requires=[str(x.req) for x in parse_requirements('requirements.txt', session='dummy')],
)
