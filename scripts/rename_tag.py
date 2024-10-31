import fire

from pytabkit.bench.data.paths import Paths
from pytabkit.models import utils


def rename_tag(old_name: str, new_name: str):
    paths = Paths.from_env_variables()
    for alg_path in paths.algs().iterdir():
        tags_path = alg_path / 'tags.yaml'
        if utils.existsFile(tags_path):
            tags = utils.deserialize(tags_path, use_yaml=True)
            tags = [tag if tag != old_name else new_name for tag in tags]
            utils.serialize(tags_path, tags, use_yaml=True)


if __name__ == '__main__':
    fire.Fire(rename_tag)
