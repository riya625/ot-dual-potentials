"""A small DirectML-friendly recurrent model for OT source potentials."""

import torch
import torch.nn as nn


class CoordinateRNN(nn.Module):
    """Vanilla RNN: h_t = tanh(W_x [X_t, Y_t] + W_h h_(t-1))."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, sequence):
        batch_size, n_steps, _ = sequence.shape
        state = sequence.new_zeros(batch_size, self.hidden_layer.in_features)
        states = []
        for step in range(n_steps):
            state = torch.tanh(self.input_layer(sequence[:, step]) + self.hidden_layer(state))
            states.append(state)
        return torch.stack(states, dim=1)


class OTPotentialModel(nn.Module):
    """Predict source potentials, optionally with target dual potentials.

    The updated training data contains X, Y, and u_star.  At each recurrent
    step the model reads one source/target coordinate pair and predicts one
    source potential.  The final centering matches the gauge convention used
    by ``ot_solve.normalize_potentials``: mean(u_star) = 0.
    """

    def __init__(self, n_dim, hidden_dim=64, predict_duals=False):
        super().__init__()
        self.encoder = CoordinateRNN(input_dim=2 * n_dim, hidden_dim=hidden_dim)
        self.output_head = nn.Linear(hidden_dim, 1)
        self.predict_duals = predict_duals
        if predict_duals:
            self.target_output_head = nn.Linear(hidden_dim, 1)

    def forward(self, source_coords, target_coords):
        if source_coords.ndim != 3 or target_coords.ndim != 3:
            raise ValueError("source_coords and target_coords must have shape (batch, n_points, n_dim)")
        if source_coords.shape != target_coords.shape:
            raise ValueError("source_coords and target_coords must have the same shape")

        sequence = torch.cat((source_coords, target_coords), dim=-1)
        hidden_states = self.encoder(sequence)
        u = self.output_head(hidden_states).squeeze(-1)
        u = u - u.mean(dim=1, keepdim=True)
        if not self.predict_duals:
            return u

        # v_star already has the gauge induced by mean(u_star) = 0, so it is
        # not centered independently.
        v = self.target_output_head(hidden_states).squeeze(-1)
        return u, v
