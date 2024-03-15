import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "cnn-classifier"
AUTHOR_USER_NAME = "AlRashidIssa"
SRC_REPO = "src"
AUTHOR_EMAIL = "rashidissa2001@outlook.sa"

setuptools.setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small Python package for CNNs application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/MLOps",
    project_urls={
        "Bug Tracker": f"https://github.com/AlRashidIssa/MLOps/tree/main/end_to_end_project",
        "Source Code": f"https://github.com/AlRashidIssa/MLOps/tree/main/end_to_end_project/src",
    },
    package_dir={"": SRC_REPO},
    packages=setuptools.find_packages(where=SRC_REPO),
)
