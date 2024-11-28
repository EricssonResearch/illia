# Libraries
from setuptools import setup, find_packages


def parse_requirements(filename: str):
    with open(filename, "r") as file:
        return file.read().splitlines()


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="illia",
    version="0.1",
    license="MIT",
    description="__THIS_IS_USED_FOR_DEVELOPMENT_ONLY__",
    long_description=readme(),
    url="",
    author="Oscar Llorente",
    author_email="oscar.llorente.gonzalez@ericsson.com",
    maintainer="Anubhab Samal, Daniel Bazo, Lucia Ferrer",
    maintainer_email="anubhab.samal@ericsson.com, dani.bazo@ericsson.com, lucia.ferrer@ericsson.com",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    include_package_data=True,
    package_data={"": ["*.zip"]},
    zip_safe=False,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python",
        "Intended Audience :: Developers",
    ],
)
