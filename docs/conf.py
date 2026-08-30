from pathlib import Path
import tomllib


project = "MLPipelineHolder"
author = "Xiaokai Cui"

with (Path(__file__).parents[1] / "pyproject.toml").open("rb") as pyproject_file:
    release = tomllib.load(pyproject_file)["project"]["version"]
version = release

extensions = []

html_title = "MLPipelineHolder Documentation"

# Copy the existing standalone API reference into the built documentation.
html_extra_path = [
    "api_reference.html",
]
