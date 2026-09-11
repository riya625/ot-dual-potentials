# Transformer for learning Kantorovich dual potentials.

#In short: an encoder-only, positional-encoding-free transformer over the concatenated source+target point
#sequence, read out only at the source tokens. This construction makes
#the output permutation-equivariant in source-point order and
#permutation-invariant in target-point order.


import torch
import torch.nn as nn


class DualPotentialTransformer(nn.Module):
    def __init__(self, input_dim=5, d_model=64, nhead=4, num_layers=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed = nn.Linear(input_dim, d_model)
        self.type_embed = nn.Embedding(2, d_model)  # 0 = source, 1 = target

        # Normalization stats (set from training data via set_normalization).
        # Coordinates are standardized before embedding; the standardized target
        # scale is folded back into the output so forward() returns u_hat in the
        # original (raw) units. Stored as buffers so they travel with the
        # state_dict and inference code needs no separate setup.
        self.register_buffer("coord_mean", torch.zeros(input_dim))
        self.register_buffer("coord_std", torch.ones(input_dim))
        self.register_buffer("u_std", torch.ones(()))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

    def set_normalization(self, coord_mean, coord_std, u_std):
        """Set input/target normalization stats (computed from training data).

        coord_mean, coord_std: shape (input_dim,) — per-axis coordinate stats
            over the pooled source+target points of the training set.
        u_std: scalar — std of the (already gauge-fixed, mean-0) u_star targets.
        """
        self.coord_mean.copy_(torch.as_tensor(coord_mean).reshape(self.input_dim))
        self.coord_std.copy_(torch.as_tensor(coord_std).reshape(self.input_dim))
        self.u_std.copy_(torch.as_tensor(u_std).reshape(()))

    def forward(self, X, Y):
        """
        X: (batch_size, n, input_dim) source point coordinates
        Y: (batch_size, n, input_dim) target point coordinates
        returns: (batch_size, n) predicted u_hat, one scalar per source point,
                 in the same (raw) units as u_star
        """
        n = X.shape[1]
        X = (X - self.coord_mean) / self.coord_std
        Y = (Y - self.coord_mean) / self.coord_std
        src = self.embed(X) + self.type_embed.weight[0]
        tgt = self.embed(Y) + self.type_embed.weight[1]
        tokens = torch.cat([src, tgt], dim=1)  # (B, 2n, d_model)
        out = self.encoder(tokens)  # (B, 2n, d_model)
        u_std_hat = self.head(out[:, :n, :]).squeeze(-1)  # (B, n), standardized
        return u_std_hat * self.u_std  # back to raw units


if __name__ == "__main__":
    torch.manual_seed(0)
    model = DualPotentialTransformer()
    model.eval()  # disable dropout so the symmetry checks are deterministic

    X = torch.randn(8, 100, 5)
    Y = torch.randn(8, 100, 5)

    # 1. Shape check
    u_hat = model(X, Y)
    assert u_hat.shape == (8, 100), u_hat.shape
    print("shape check: passed", tuple(u_hat.shape))

    # 2. Gradient flow check
    model.zero_grad()
    loss = model(X, Y).sum()
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters with no grad: {missing}"
    print("gradient flow check: passed (all", sum(1 for _ in model.parameters()), "params have grad)")

    # 3. Permutation-equivariance in source order
    with torch.no_grad():
        base = model(X, Y)
        perm = torch.randperm(X.shape[1])
        permuted = model(X[:, perm, :], Y)
        assert torch.allclose(permuted, base[:, perm], atol=1e-5), \
            (permuted - base[:, perm]).abs().max().item()
    print("source permutation-equivariance check: passed")

    # 4. Permutation-invariance in target order
    with torch.no_grad():
        tperm = torch.randperm(Y.shape[1])
        invariant = model(X, Y[:, tperm, :])
        assert torch.allclose(invariant, base, atol=1e-5), \
            (invariant - base).abs().max().item()
    print("target permutation-invariance check: passed")
