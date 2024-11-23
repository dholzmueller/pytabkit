from typing import Optional

import fire

from pytabkit.models import utils
from scripts.move_algs import move_algs


def move_many_algs(base_path_1: str, base_path_2: str, algs_filename: Optional[str] = None, prefixes_filename: Optional[str] = None,
                   dry_run: bool = False):
    if algs_filename is None:
        algs = []
    else:
        algs = [name.strip() for name in utils.readFromFile(algs_filename).split('\n') if name.strip() != '']

    if prefixes_filename is None:
        prefixes = []
    else:
        pprefixes = [name.strip() for name in utils.readFromFile(prefixes_filename).split('\n') if name.strip() != '']

    move_algs(base_path_1, base_path_2, *algs, dry_run=dry_run)
    for prefix in prefixes:
        move_algs(base_path_1, base_path_2, startswith=prefix, dry_run=dry_run)


if __name__ == '__main__':
    fire.Fire(move_many_algs)
