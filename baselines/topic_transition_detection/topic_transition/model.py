from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .common import CHANNELS, RATES, modalities


class CausalBlock(nn.Module):
    def __init__(self, hidden, kernel, dilation, dropout):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.layers = nn.ModuleList([nn.Conv1d(hidden, hidden, kernel, dilation=dilation) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x
        for layer in self.layers:
            x = self.dropout(F.relu(layer(F.pad(x, (self.padding, 0))))) * mask
        return F.relu(x + residual) * mask


class TemporalEncoder(nn.Module):
    def __init__(self, modality, config):
        super().__init__()
        hidden = config["hidden"]
        # Explicit validity channels distinguish missing input from a real zero.
        self.projection = nn.Conv1d(CHANNELS[modality] * 2, hidden, 1)
        self.blocks = nn.ModuleList([CausalBlock(hidden, config["kernel_size"], d, config["dropout"])
                                     for d in config["dilations"][modality]])
        self.phase_bins = config.get("phase_bins")
        self.bin_steps = None if self.phase_bins else int(2.5 * RATES[modality])
        self.receptive_field = 1 + 2 * (config["kernel_size"] - 1) * sum(config["dilations"][modality])

    def forward(self, values, feature_mask, bin_ids=None):
        b, t, p, c = values.shape
        x = torch.where(feature_mask, values, 0).permute(0, 2, 3, 1).reshape(b * p, c, t)
        fm = feature_mask.permute(0, 2, 3, 1).reshape(b * p, c, t)
        mask = fm.any(dim=1, keepdim=True).to(x.dtype)
        x = self.projection(torch.cat([x, fm.to(x.dtype)], dim=1)) * mask
        for layer in self.blocks:
            x = layer(x, mask)
        if self.phase_bins:
            if bin_ids is None or bin_ids.shape != (b, t):
                raise ValueError("Variable-window TCN requires per-timestep phase bins")
            x = x.reshape(b, p, -1, t)
            mask = mask.reshape(b, p, 1, t)
            vectors, valid = [], []
            for index in range(len(self.phase_bins)):
                where = mask * (bin_ids[:, None, None, :] == index)
                vectors.append((x * where).sum(-1) / where.sum(-1).clamp_min(1))
                valid.append(where.sum(-1).squeeze(2) > 0)
            return torch.stack(vectors, dim=2), torch.stack(valid, dim=2)
        x = x.reshape(b, p, -1, 6, self.bin_steps)
        mask = mask.reshape(b, p, 1, 6, self.bin_steps)
        pooled = (x * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        return pooled.permute(0, 1, 3, 2), mask.sum(-1).squeeze(2) > 0


class TransitionTCN(nn.Module):
    def __init__(self, modality, config, group=False):
        super().__init__()
        self.group = group
        self.n_bins = len(config["phase_bins"]) if config.get("phase_bins") else 6
        self.encoders = nn.ModuleDict({m: TemporalEncoder(m, config) for m in modalities(modality)})
        h, d = config["hidden"], config["dropout"]
        person_dim = h * len(self.encoders)
        if group:
            self.relation = nn.Sequential(nn.Linear(person_dim * 3, h), nn.ReLU(), nn.Dropout(d), nn.Linear(h, h), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(self.n_bins * (person_dim + (2 * h if group else 0)), h), nn.ReLU(),
                                        nn.Dropout(d), nn.Linear(h, 1))

    def forward(self, batch):
        embedded, valid = [], []
        for m, encoder in self.encoders.items():
            x, mask = batch[m], batch[m + "_mask"]
            mask = mask & batch["person_present"][:, None, :, None]
            if not self.group:
                x, mask = x[:, :, :1], mask[:, :, :1]
            z, vm = encoder(x, mask, batch.get(m + "_bin_ids"))
            embedded.append(z)
            valid.append(vm)
        z = torch.cat(embedded, dim=-1)  # B, P, bin, channel: modalities fuse within person/bin.
        focal = z[:, 0]
        pieces = [focal]
        if self.group:
            recipients = z[:, 1:]
            anchor = focal[:, None].expand_as(recipients)
            relations = self.relation(torch.cat([recipients, anchor, recipients - anchor], dim=-1))
            mask = torch.stack(valid).any(0)[:, 1:] & batch["person_present"][:, 1:, None]
            mask = mask[..., None]
            mean = (relations * mask).sum(1) / mask.sum(1).clamp_min(1)
            maximum = relations.masked_fill(~mask, -torch.inf).amax(1)
            maximum = torch.where(mask.any(1), maximum, 0)
            pieces.extend([mean, maximum])
        return self.classifier(torch.cat(pieces, dim=-1).flatten(1)).squeeze(-1)
