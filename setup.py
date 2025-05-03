from setuptools import find_packages, setup

setup(
    name='src',
    version='0.1',
    description='A short description of the project.',
    author='chirag mohit',
    license='',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)
