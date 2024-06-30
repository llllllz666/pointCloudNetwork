import yaml
from datasets.cloth_synthetic import ClothSynthetic
from core.models.equi_contrast import EquiContrast
from core.trainer import Trainer

dataset_dict = {
    "cloth_synthetic": ClothSynthetic,
}

model_dict = {
    "equi_contrast": EquiContrast,
}

def get_trainer(model, optimizer, cfg, device):
    """ Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    """
    trainer = Trainer(model, optimizer, cfg, device)
    return trainer

def get_dataset(cfg, mode):
    """ Returns the dataset.

    Args:
        cfg (dict): config dictionary
        mode: dataset mode
    """

    dataset = dataset_dict[cfg["data"]["dataset_class"]](cfg["data"]["dataset"], mode=mode)

    return dataset

def get_model(cfg, device="cuda"):
    """ Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    """
    model = model_dict[cfg["model"]["model_name"]](cfg["model"]).to(device)
    return model