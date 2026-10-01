"""Train and evaluate a simple RNN on the dataset written by ot_solve.py.

Create the data first, for example:
    python ot_solve.py --num_instances 300 --out_dir new
Then run:
    python pipeline.py
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ot_sparsity import OTPotentialModel


class OTDataset(Dataset):
    """The solver's point clouds and dual-potential targets from one .npz file."""

    def __init__(self, path, include_v=False):
        with np.load(path) as data:
            required = {"X", "Y", "u_star"}
            if include_v:
                required.add("v_star")
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{path} is missing required arrays: {sorted(missing)}")
            self.sources = torch.from_numpy(data["X"]).float()
            self.targets = torch.from_numpy(data["Y"]).float()
            self.u_star = torch.from_numpy(data["u_star"]).float()
            self.v_star = torch.from_numpy(data["v_star"]).float() if include_v else None

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        if self.v_star is None:
            return self.sources[index], self.targets[index], self.u_star[index]
        return self.sources[index], self.targets[index], self.u_star[index], self.v_star[index]


def get_device():
    """Use DirectML when installed; otherwise run on CPU."""
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


def batch_loss(model, batch, device, criterion):
    if model.predict_duals:
        source, target, true_u, true_v = (item.to(device) for item in batch)
        predicted_u, predicted_v = model(source, target)
        return criterion(predicted_u, true_u) + criterion(predicted_v, true_v)

    source, target, true_u = (item.to(device) for item in batch)
    return criterion(model(source, target), true_u)


def run_epoch(model, loader, device, criterion, optimizer=None):
    """Train if optimizer is supplied; otherwise evaluate the held-out split."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for batch in loader:
            loss = batch_loss(model, batch, device, criterion)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)


def plot_test_predictions(model, loader, device, filename):
    """Save a scatter and line comparison of u prediction against u_star."""
    import matplotlib.pyplot as plt

    model.eval()
    batch = next(iter(loader))
    source, target, true_u = batch[:3]
    with torch.no_grad():
        output = model(source.to(device), target.to(device))
    predicted_u = output[0].cpu() if model.predict_duals else output.cpu()

    true_values = true_u.numpy().ravel()
    predicted_values = predicted_u.numpy().ravel()
    low = min(true_values.min(), predicted_values.min())
    high = max(true_values.max(), predicted_values.max())

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(true_values, predicted_values, s=8, alpha=0.35)
    axes[0].plot([low, high], [low, high], "k--", label="perfect prediction")
    axes[0].set(title="Held-out source potentials", xlabel="True u_star", ylabel="Predicted u")
    axes[0].legend()

    point_index = np.arange(true_u.shape[1])
    axes[1].plot(point_index, true_u[0].numpy(), label="true u_star")
    axes[1].plot(point_index, predicted_u[0].numpy(), "--", label="predicted u")
    axes[1].set(title="First held-out OT problem", xlabel="Source point index", ylabel="Potential")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(filename, dpi=160)
    plt.close(figure)
    print(f"Saved prediction plot to {filename}")


def plot_loss(training_losses, test_losses, filename):
    """Save train and held-out MSE over epochs."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(training_losses) + 1)
    axis.plot(epochs, training_losses, marker="o", markersize=3, label="train")
    axis.plot(epochs, test_losses, marker="o", markersize=3, label="test")
    axis.set(title="RNN loss", xlabel="Epoch", ylabel="MSE")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(filename, dpi=160)
    plt.close(figure)
    print(f"Saved loss plot to {filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="new")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument(
        "--predict_duals", action="store_true",
        help="Train source and target potential heads using u_star and v_star.",
    )
    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, "train.npz")
    test_path = os.path.join(args.data_dir, "test.npz")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Expected {train_path} and {test_path}. Run ot_solve.py first to create them."
        )

    train_set = OTDataset(train_path, include_v=args.predict_duals)
    test_set = OTDataset(test_path, include_v=args.predict_duals)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    device = get_device()
    model = OTPotentialModel(
        n_dim=train_set.sources.shape[-1],
        hidden_dim=args.hidden_dim,
        predict_duals=args.predict_duals,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    training_losses, test_losses = [], []
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, device, criterion, optimizer)
        test_loss = run_epoch(model, test_loader, device, criterion)
        training_losses.append(train_loss)
        test_losses.append(test_loss)
        print(
            f"Epoch {epoch + 1:2d}/{args.epochs}: "
            f"train MSE = {train_loss:.4f}, test MSE = {test_loss:.4f}"
        )

    os.makedirs("new", exist_ok=True)
    plot_loss(training_losses, test_losses, "new/loss_curve.png")
    plot_test_predictions(model, test_loader, device, "new/test_potential_predictions.png")
    torch.save(model.state_dict(), "new/ot_potential_rnn.pth")


if __name__ == "__main__":
    main()
