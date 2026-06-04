from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import DomainConfig, ModelConfig, PDEConfig, SeedConfig


def _inv_softplus(value: float) -> float:
    return math.log(math.exp(value) - 1.0)


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, features: int, sigma: float) -> None:
        super().__init__()
        self.register_buffer("basis", torch.randn(features, in_dim) * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = 2.0 * math.pi * x @ self.basis.T
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


def square_geo_features(xy: torch.Tensor, box: float) -> torch.Tensor:
    scaled = (xy / box).clamp(0.0, 1.0)
    x = scaled[:, 0:1]
    y = scaled[:, 1:2]
    distances = torch.cat([x, 1.0 - x, y, 1.0 - y], dim=-1)
    boundary_distance = torch.min(distances, dim=-1, keepdim=True).values
    centered = scaled - 0.5
    radius = torch.linalg.norm(centered, dim=-1, keepdim=True)
    interior_mode = torch.sin(math.pi * x) * torch.sin(math.pi * y)
    interaction = x * y
    return torch.cat([distances, boundary_distance, radius, interior_mode, interaction], dim=-1)


class GatedMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, out_dim: int) -> None:
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden)
        self.u_layer = nn.Linear(in_dim, hidden)
        self.v_layer = nn.Linear(in_dim, hidden)
        self.hidden_layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.out_layer = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.in_layer(x))
        u = torch.tanh(self.u_layer(x))
        v = torch.tanh(self.v_layer(x))
        for layer in self.hidden_layers:
            gate = torch.sigmoid(layer(h))
            h = (1.0 - gate) * u + gate * v
        return self.out_layer(h)


class SourceHead(nn.Module):
    def __init__(self, domain: DomainConfig, seed: SeedConfig) -> None:
        super().__init__()
        self.box = float(domain.box)
        # The inverse model must not be initialized at the synthetic truth origin.
        # Start from a neutral middle-of-domain source and let data + physics move it.
        cx = torch.tensor(0.5).clamp(1.0e-4, 1.0 - 1.0e-4)
        cy = torch.tensor(0.5).clamp(1.0e-4, 1.0 - 1.0e-4)
        self.center_logits = nn.Parameter(torch.logit(torch.stack([cx, cy])))
        init_sigma = max(0.08 * domain.box, seed.sigma)
        self.raw_sigma = nn.Parameter(torch.tensor(_inv_softplus(init_sigma)))
        self.raw_amplitude = nn.Parameter(torch.tensor(_inv_softplus(0.35)))

    def center(self) -> torch.Tensor:
        return torch.sigmoid(self.center_logits) * self.box

    def set_center(self, center_xy: tuple[float, float]) -> None:
        clipped = torch.tensor(center_xy, dtype=self.center_logits.dtype, device=self.center_logits.device)
        clipped = (clipped / self.box).clamp(1.0e-4, 1.0 - 1.0e-4)
        with torch.no_grad():
            self.center_logits.copy_(torch.logit(clipped))

    def sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma).clamp(0.01, 0.25)

    def amplitude(self) -> torch.Tensor:
        return F.softplus(self.raw_amplitude).clamp(0.01, 1.0)

    def profile(self, xy: torch.Tensor) -> torch.Tensor:
        center = self.center().view(1, 2)
        sigma = self.sigma()
        dist2 = ((xy - center) ** 2).sum(dim=-1, keepdim=True)
        return self.amplitude() * torch.exp(-dist2 / (2.0 * sigma**2))


class PDEParameters(nn.Module):
    def __init__(self, pde: PDEConfig, model: ModelConfig) -> None:
        super().__init__()
        self.include_advection = bool(pde.include_advection)
        self.raw_diffusion = nn.Parameter(
            torch.tensor(_inv_softplus(pde.diffusion)),
            requires_grad=model.learn_diffusion,
        )
        self.raw_reaction = nn.Parameter(
            torch.tensor(_inv_softplus(pde.reaction)),
            requires_grad=model.learn_reaction,
        )
        self.velocity = nn.Parameter(
            torch.tensor([pde.velocity_x, pde.velocity_y], dtype=torch.float32),
            requires_grad=model.learn_drift,
        )

    def diffusion(self) -> torch.Tensor:
        return F.softplus(self.raw_diffusion)

    def reaction(self) -> torch.Tensor:
        return F.softplus(self.raw_reaction)


class OriginPINN(nn.Module):
    def __init__(self, domain: DomainConfig, pde: PDEConfig, seed: SeedConfig, model: ModelConfig) -> None:
        super().__init__()
        self.domain = domain
        self.use_source_envelope = bool(model.use_source_envelope)
        self.use_geo_features = bool(model.use_geo_features)
        self.spatial_fourier_only = bool(model.spatial_fourier_only)
        fourier_dim = 2 if self.spatial_fourier_only else 3
        self.features = FourierFeatures(fourier_dim, model.fourier_features, model.fourier_sigma)
        geo_dim = 8 if self.use_geo_features else 0
        self.mlp = GatedMLP(2 * model.fourier_features + 3 + geo_dim, model.hidden, model.layers, 1)
        self.source = SourceHead(domain, seed)
        self.pde = PDEParameters(pde, model)

    def forward(self, xy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xyt = torch.cat([xy / self.domain.box, t / self.domain.t_end], dim=-1)
        spectral_input = xyt[:, :2] if self.spatial_fourier_only else xyt
        pieces = [xyt, self.features(spectral_input)]
        if self.use_geo_features:
            pieces.append(square_geo_features(xy, self.domain.box))
        z = torch.cat(pieces, dim=-1)
        neural_field = torch.sigmoid(self.mlp(z))
        if not self.use_source_envelope:
            return neural_field
        source_field = self.source.profile(xy).clamp(0.0, 1.0)
        # A hard initial-source envelope: exactly source-dominated at t=0, but
        # negligible in the late observation window. This gives source parameters
        # gradients through the PDE residual instead of treating origin extraction
        # as a post-hoc argmax.
        blend = torch.exp(-t / (0.08 * self.domain.t_end + 1.0e-8))
        return blend * source_field + (1.0 - blend) * neural_field

    def sparse_last_layer_l1(self) -> torch.Tensor:
        return self.mlp.out_layer.weight.abs().mean()

    def physics_dict(self) -> dict[str, float]:
        return {
            "diffusion": float(self.pde.diffusion().detach().cpu()),
            "reaction": float(self.pde.reaction().detach().cpu()),
            "velocity_x": float(self.pde.velocity[0].detach().cpu()),
            "velocity_y": float(self.pde.velocity[1].detach().cpu()),
            "source_x": float(self.source.center()[0].detach().cpu()),
            "source_y": float(self.source.center()[1].detach().cpu()),
            "source_sigma": float(self.source.sigma().detach().cpu()),
            "source_amplitude": float(self.source.amplitude().detach().cpu()),
        }
