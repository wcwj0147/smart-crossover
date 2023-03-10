from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent.parent


def get_data_dir_path() -> Path:
    return get_project_root() / "data"
