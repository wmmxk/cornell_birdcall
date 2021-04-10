from config_params.example_config import Parameters
from models.sed_models import PANNsDense121Att


def test_model():
    hparams = Parameters()
    model = PANNsDense121Att(**hparams.model_config)
    print(model)


test_model()
