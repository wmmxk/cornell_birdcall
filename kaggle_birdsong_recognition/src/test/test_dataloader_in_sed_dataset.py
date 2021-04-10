# from augmentations.sed_default_augment import get_transforms
from augmentations.sed_background_augment import get_transforms
from config_params.example_config import Parameters
from dataloaders.sed_dataset import SedDataset
from torch.utils.data import DataLoader
from ignite.utils import convert_tensor
from models.sed_models import PANNsDense121Att
import ignite.distributed as idist
from loss.sed_scaled_pos_neg_focal_loss import SedScaledPosNegFocalLoss


def test_dataloader():
    hparams = Parameters()
    transforms = get_transforms(bckgrd_aug_dir=hparams.bckgrd_aug_dir, secondary_bckgrd_aug_dir=hparams.secondary_bckgrd_aug_dir)
    train_ds = SedDataset(**hparams.train_ds_params, transform=transforms['train'])
    sampler = train_ds.sampler(train_ds, train_ds.get_label)
    kwargs = {"batch_size": hparams.train_bs, "num_workers": hparams.train_num_workers, "shuffle": False, "drop_last":True,
              "sampler": sampler, "worker_init_fn": train_ds.additional_loader_params['worker_init_fn'], 'pin_memory': True}

    train_loader = DataLoader(train_ds, **kwargs)
    dataloader_iter = iter(train_loader)
    for i in train_ds:
        print(i.keys())
        break
    batch = next(dataloader_iter)
    # prepare_batch method of SedEngine
    device = idist.device()
    x = convert_tensor(batch["waveforms"], device=device, non_blocking=True)
    bs, c, s = x.shape
    y_target = {}
    all_labels = convert_tensor(batch["all_labels"], device=device, non_blocking=True)
    y_target["all_labels"] = all_labels
    primary_labels = convert_tensor(batch["primary_labels"], device=device, non_blocking=True)
    secondary_labels = convert_tensor(batch["secondary_labels"], device=device, non_blocking=True)
    y_target["secondary_labels"] = secondary_labels

    mixup_lambda = None
    x, y = ((x, mixup_lambda), {"all_labels": all_labels, "primary_labels": primary_labels, "secondary_labels": secondary_labels})

    # create model
    model = PANNsDense121Att(**hparams.model_config).to("cuda")
    model.train()
    y_pred = model(x)
    criterion = SedScaledPosNegFocalLoss(**hparams.criterion_params)
    print(y)

test_dataloader()