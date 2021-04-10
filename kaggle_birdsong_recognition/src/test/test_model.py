from models.sed_models import PANNsDense121Att

from config_params.example_config import Parameters
from dataloaders.sed_dataset import SedDataset
from torch.utils.data import DataLoader


def test_model():
    hparams = Parameters()
    model = PANNsDense121Att(**hparams.model_config)
    print(model)
    print("good"*100)


test_model()