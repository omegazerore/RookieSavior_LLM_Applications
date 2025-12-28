import sys
import os


def get_project_dir() -> str:

    """
    Get the full path to the repository
    """

    platform = sys.platform

    if platform == 'linux':
        dir_as_list = os.path.dirname(__file__).split("/")
    else:
        dir_as_list = os.path.dirname(__file__).split("\\")
    index = dir_as_list.index("src")
    project_directory = "/".join(dir_as_list[:index])

    return project_directory


def get_file(relative_path: str) -> str:

    """
    Given the relative path to the repository, return the full path
    Args:
        relative_path: relative path of file to the project directory
    """

    return os.path.join(get_project_dir(), relative_path)


def get_datafetch():

    try:
        datafetch = os.environ['TARGET_DIR']
    except KeyError:
        datafetch = os.path.join(get_project_dir(), 'data', 'datafetch')
    if not os.path.isdir(datafetch):
        os.makedirs(datafetch)

    return datafetch
