#Train DualPotentialTransformer to regress u_hat against u_star.

#Loads the saved train/test data, trains the model, tracks train/test MSE loss
#and R^2 per epoch, and saves a checkpoint, loss-curve and R^2-curve plots,
#and a CSV of the per-epoch numbers those plots are drawn from.



import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import DualPotentialTransformer

# --- Config -----------------------------------------------------------------
SEED = 0
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
GRAD_CLIP_NORM = 1.0  # transformer training is unstable at 5D without this;
                      # set to None to disable
NUM_EPOCHS = 100  # sanity-check value for the 300-instance dataset; revisit
                  # after seeing the loss curve / regenerating with more data
TRAIN_PATH = "data/train.npz"
TEST_PATH = "data/test.npz"
CHECKPOINT_PATH = "checkpoints/model.pth"
LOSS_PLOT_PATH = "results/loss_curve.png"
R2_PLOT_PATH = "results/r2_curve.png"
METRICS_PATH = "results/metrics.csv"


class PointCloudDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data["X"]).float()
        self.Y = torch.from_numpy(data["Y"]).float()
        self.u_star = torch.from_numpy(data["u_star"]).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.u_star[idx]


def compute_norm_stats(dataset):
    """Per-axis coordinate mean/std (pooled source+target) and u_star std,
    computed from the training set only. These are handed to the model so it
    standardizes its inputs and de-standardizes its output internally."""
    dim = dataset.X.shape[-1]
    coords = torch.cat([dataset.X, dataset.Y], dim=1).reshape(-1, dim)
    coord_mean = coords.mean(dim=0)
    coord_std = coords.std(dim=0)
    u_std = dataset.u_star.std()
    return coord_mean, coord_std, u_std


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for X, Y, u_star in loader:
        X, Y, u_star = X.to(device), Y.to(device), u_star.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X, Y), u_star)
        loss.backward()
        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()


def evaluate_metrics(model, loader, device):
    """Clean pass over a split: per-point MSE and R^2.

    R^2 = 1 - SS_res / SS_tot, where SS_tot is measured against the split's own
    u_star mean — i.e. MSE normalized by Var(u_star), so R^2=0 means "no better
    than predicting the mean" and R^2=1 means "perfect". Sums are pooled over
    every (instance, point) pair, so uneven batch sizes don't bias it.
    """
    model.eval()  # disable dropout for stable, comparable numbers
    total_se = 0.0   # sum of (u_hat - u_star)^2
    total_u = 0.0    # sum of u_star
    total_u2 = 0.0   # sum of u_star^2
    count = 0
    with torch.no_grad():
        for X, Y, u_star in loader:
            X, Y, u_star = X.to(device), Y.to(device), u_star.to(device)
            u_hat = model(X, Y)
            total_se += ((u_hat - u_star) ** 2).sum().item()
            total_u += u_star.sum().item()
            total_u2 += (u_star ** 2).sum().item()
            count += u_star.numel()
    mse = total_se / count
    ss_tot = total_u2 - total_u ** 2 / count
    r2 = 1.0 - total_se / ss_tot
    return mse, r2


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = PointCloudDataset(TRAIN_PATH)
    test_ds = PointCloudDataset(TEST_PATH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"train instances: {len(train_ds)}, test instances: {len(test_ds)}")

    input_dim = train_ds.X.shape[-1]
    model = DualPotentialTransformer(input_dim=input_dim).to(device)
    coord_mean, coord_std, u_std = compute_norm_stats(train_ds)
    model.set_normalization(coord_mean, coord_std, u_std)
    print(
        f"norm stats (from train): coord_mean={coord_mean.tolist()}, "
        f"coord_std={coord_std.tolist()}, u_std={u_std.item():.3f}"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    train_losses, test_losses = [], []
    train_r2s, test_r2s = [], []
    for epoch in range(NUM_EPOCHS):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_loss, train_r2 = evaluate_metrics(model, train_loader, device)
        test_loss, test_r2 = evaluate_metrics(model, test_loader, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_r2s.append(train_r2)
        test_r2s.append(test_r2)
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}  "
            f"train_mse={train_loss:.4f} test_mse={test_loss:.4f}  "
            f"train_r2={train_r2:.4f} test_r2={test_r2:.4f}"
        )

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"saved checkpoint to {CHECKPOINT_PATH}")

    os.makedirs(os.path.dirname(LOSS_PLOT_PATH), exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.savefig(LOSS_PLOT_PATH)
    print(f"saved loss curve to {LOSS_PLOT_PATH}")

    plt.figure()
    plt.plot(train_r2s, label="train")
    plt.plot(test_r2s, label="test")
    plt.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    plt.xlabel("epoch")
    plt.ylabel(r"$R^2$  (1 - MSE / Var($u^\star$))")
    plt.ylim(top=1.0)
    plt.legend()
    plt.savefig(R2_PLOT_PATH)
    print(f"saved R^2 curve to {R2_PLOT_PATH}")

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mse", "test_mse", "train_r2", "test_r2"])
        for epoch, (tl, el, tr, er) in enumerate(
            zip(train_losses, test_losses, train_r2s, test_r2s), start=1
        ):
            writer.writerow([epoch, tl, el, tr, er])
    print(f"saved per-epoch metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
