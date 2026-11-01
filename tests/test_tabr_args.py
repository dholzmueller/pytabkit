import pytest

from pytabkit.models.nn_models.tabr import TabrModel
from pytabkit.models.nn_models.tabr_context_freeze import TabrModelContextFreeze

# These checks run before faiss is imported, so they do not need faiss.
ARGS = dict(n_num_features=3, n_bin_features=0, cat_cardinalities=[], n_classes=2, num_embeddings=None,
            d_main=8, d_multiplier=2.0, encoder_n_blocks=0, predictor_n_blocks=1, mixer_normalization='auto',
            context_dropout=0.0, dropout0=0.0, dropout1='dropout0', normalization='LayerNorm', activation='ReLU')


@pytest.mark.parametrize("model_class", [TabrModel, TabrModelContextFreeze])
def test_candidate_encoding_batch_size_needs_memory_efficient(model_class):
    with pytest.raises(ValueError, match="memory_efficient=True"):
        model_class(**ARGS, memory_efficient=False, candidate_encoding_batch_size=1000)
