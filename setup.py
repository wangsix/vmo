import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vmo",
    version="0.30.4",
    author="Cheng-i Wang",
    author_email="chw160@ucsd.edu",
    description="Variable Markov Oracle in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wangsix/vmo",
    project_urls={
        "Bug Tracker": "https://github.com/wangsix/vmo/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=['vmo', 'vmo.VMO', 'vmo.analysis', 'vmo.realtime', 'vmo.VMO.utility', 'vmo.VMO.utility.distances'],
    python_requires=">=3.7",
)
