import os
import numpy as np
import torch
from torch import optim
from tensorboardX import SummaryWriter
import argparse

from utils import config
from utils.checkpoints import CheckpointIO

parser = argparse.ArgumentParser(description="Train the point feature extraction model.")
parser.add_argument("--config", default="configs/default.yaml", type=str, help="Path to config file.")
args = parser.parse_args()

cfg = config.load_config(args.config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

out_dir = cfg["training"]["out_dir"]
batch_size = cfg["training"]["batch_size"]
batch_size_vis = cfg["training"]["batch_size_vis"]
batch_size_val = cfg["training"]["batch_size_val"]
backup_every = cfg["training"]["backup_every"]
lr = cfg["training"]["learning_rate"]

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Dataset
train_dataset = config.get_dataset(cfg, "train")
val_dataset = config.get_dataset(cfg, "val")

# Dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_val, num_workers=4, shuffle=False, drop_last=True)
# For visualizations
vis_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_vis, shuffle=False, drop_last=True)

# Model
model = config.get_model(cfg, device=device)

# Get optimizer and trainer
optimizer = optim.Adam(model.parameters(), lr=lr)
trainer = config.get_trainer(model, optimizer, cfg, device=device)

# Load pre-trained model is existing
kwargs = {
    "model": model,
    "optimizer": optimizer,
}
checkpoint_io = CheckpointIO(
    out_dir, initialize_from=cfg["training"]["initialize_from"],
    initialization_file_name=cfg["training"]["initialization_file_name"],
    **kwargs)
try:
    load_dict = checkpoint_io.load("model.pt")
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get("epoch_it", -1)
it = load_dict.get("it", -1)
metric_val_best = load_dict.get(
    'loss_val_best', np.inf)

logger = SummaryWriter(os.path.join(out_dir, "logs"))

# Shorthands
print_every = cfg["training"]["print_every"]
checkpoint_every = cfg["training"]["checkpoint_every"]
validate_every = cfg["training"]["validate_every"]
visualize_every = cfg["training"]["visualize_every"]

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print(model)
print("Total number of parameters: %d" % nparameters)

def swap_batch_dim(batch):
    for key in batch:
        if type(batch[key]) == torch.Tensor:
            batch[key] = batch[key].transpose(0, 1).contiguous()
    return batch

# Training loop
#for epoch_it in range(-1, cfg["training"]["epochs"] + 1):
while True:
    epoch_it += 1
    if "epochs" in cfg["training"] and epoch_it > cfg["training"]["epochs"]:
        break

    for batch in train_loader:
        it += 1
        if "iters" in cfg["training"] and it > cfg["training"]["iters"]:
            break

        if cfg["data"]["swap_batch_dim"]:
            batch = swap_batch_dim(batch)

        loss = trainer.train_step(batch)
        logger.add_scalar("train/loss", loss, it)

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            print("[Epoch %02d] it=%03d, loss=%.4f" % (epoch_it, it, loss))

        # Visualize output
        if visualize_every > 0 and (it % visualize_every) == 0:
            print("Visualizing")
            data_vis = next(iter(vis_loader))
            if cfg["data"]["swap_batch_dim"]:
                data_vis = swap_batch_dim(data_vis)
            trainer.visualize(data_vis, it)

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print("Saving checkpoint")
            checkpoint_io.save("model.pt", epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)
        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
            eval_dict = trainer.evaluate(val_loader)
            metric_val = eval_dict["loss"]
            print("Validation metric: %.4f" % metric_val)

            for k, v in eval_dict.items():
                logger.add_scalar("val/%s" % k, v, it)

            if metric_val < metric_val_best:
                metric_val_best = metric_val
                print("New best model (loss %.4f)" % metric_val_best)
                checkpoint_io.save("model_best.pt", epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)
