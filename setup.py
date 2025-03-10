import pathlib
from setuptools import find_packages, setup


def get_version() -> str:
    rel_path = "src/araclip/__init__.py"
    with open(rel_path, "r") as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


def read_requirements(path: str) -> list[str]:
    with open(path, "r") as file:
        return [
            line.strip() for line in file if line.strip() and not line.startswith("#")
        ]


setup(
    name="araclip",
    version=get_version(),
    description="a python package for loading images",
    long_description=pathlib.Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    Homepage="https://github.com/Arabic-Clip/Araclip",
    url="https://github.com/Arabic-Clip/Araclip",
    Issues="https://github.com/Arabic-Clip/Araclip/issues",
    authors=[{"name": "Muhammad Al-Barham", "email": "mohammedbrham98@gmail.com"}],
    author_email="mohammedbrham98@gmail.com",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    classifiers=["Topic :: Utilities", "Programming Language :: Python :: 3.10"],
    install_requires=read_requirements("requirements.txt"),
)
