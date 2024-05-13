from setuptools import setup, find_packages

# define requirements
install_requires = ["numpy", "torch"]


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="illia",
    version="0.1",
    description="__THIS_IS_USED_FOR_DEVELOPMENT_ONLY__",
    long_description=readme(),
    url="",
    author="Oscar Llorente",
    author_email="oscar.llorente.gonzalez@ericsson.com",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    package_data={"": ["*.zip"]},
    zip_safe=False,
)
