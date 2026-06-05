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


class FactorizedLinear(nn.Module):
    """Linear layer with row-wise random weight factorization.

    The parameterization follows the PINN/PirateNet literature:
    W = diag(exp(s)) V. It keeps the initial effective weight equal to a
    Glorot-initialized matrix while optimizing the scale and direction
    separately.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        scale_mu: float = 0.0,
        scale_sigma: float = 0.1,
    ) -> None:
        super().__init__()
        weight = torch.empty(out_features, in_features)
        nn.init.xavier_uniform_(weight)
        scale = torch.empty(out_features).normal_(mean=scale_mu, std=scale_sigma)
        self.log_scale = nn.Parameter(scale)
        self.weight_v = nn.Parameter(weight / torch.exp(scale).view(-1, 1))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.exp(self.log_scale).view(-1, 1) * self.weight_v
        return F.linear(x, weight, self.bias)


def _linear(in_features: int, out_features: int, *, factorized: bool) -> nn.Module:
    if factorized:
        return FactorizedLinear(in_features, out_features)
    layer = nn.Linear(in_features, out_features)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


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


def seed_front_features(
    xy: torch.Tensor,
    t: torch.Tensor,
    center: torch.Tensor,
    sigma: float,
    diffusion: float,
    front_speed: float,
    box: float,
) -> torch.Tensor:
    center = center.to(device=xy.device, dtype=xy.dtype).view(1, 2)
    dist = torch.linalg.norm(xy - center, dim=-1, keepdim=True)
    speed = torch.as_tensor(front_speed, dtype=xy.dtype, device=xy.device)
    sigma_t = torch.sqrt(torch.as_tensor(sigma**2, dtype=xy.dtype, device=xy.device) + 2.0 * diffusion * t)
    expected_front = 3.0 * sigma + speed * t
    signed_front_distance = (expected_front - dist) / box
    gaussian_hint = torch.exp(-(dist**2) / (2.0 * sigma_t.clamp_min(1.0e-4) ** 2))
    radial_decay = torch.exp(-dist / max(3.0 * sigma, 1.0e-4))
    return torch.cat([dist / box, signed_front_distance, gaussian_hint, radial_decay], dim=-1)


def traveling_wave_features(
    xy: torch.Tensor,
    t: torch.Tensor,
    center: torch.Tensor,
    sigma: float,
    diffusion: torch.Tensor,
    reaction: torch.Tensor,
    box: float,
) -> torch.Tensor:
    center = center.to(device=xy.device, dtype=xy.dtype).view(1, 2)
    dist = torch.linalg.norm(xy - center, dim=-1, keepdim=True)
    diffusion = diffusion.to(dtype=xy.dtype, device=xy.device).clamp_min(1.0e-12)
    reaction = reaction.to(dtype=xy.dtype, device=xy.device).clamp_min(1.0e-12)
    speed = 2.0 * torch.sqrt(diffusion * reaction)
    thickness = torch.sqrt(diffusion / reaction).clamp_min(1.0e-4)
    front_radius = 3.0 * sigma + speed * t
    xi = ((dist - front_radius) / thickness).clamp(-8.0, 8.0)
    front_bump = torch.exp(-0.5 * xi.pow(2))
    inside_hint = torch.sigmoid(-xi)
    return torch.cat([xi / 8.0, torch.tanh(xi), inside_hint, front_bump, front_radius / box], dim=-1)


def seed_spatial_features(
    xy: torch.Tensor,
    center: torch.Tensor,
    sigma: float,
    box: float,
) -> torch.Tensor:
    center = center.to(device=xy.device, dtype=xy.dtype).view(1, 2)
    dist = torch.linalg.norm(xy - center, dim=-1, keepdim=True)
    signed_initial_front = (3.0 * sigma - dist) / box
    gaussian_hint = torch.exp(-(dist**2) / (2.0 * max(sigma, 1.0e-4) ** 2))
    radial_decay = torch.exp(-dist / max(3.0 * sigma, 1.0e-4))
    return torch.cat([dist / box, signed_initial_front, gaussian_hint, radial_decay], dim=-1)


class GatedMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, out_dim: int, *, factorized: bool = False) -> None:
        super().__init__()
        self.in_layer = _linear(in_dim, hidden, factorized=factorized)
        self.u_layer = _linear(in_dim, hidden, factorized=factorized)
        self.v_layer = _linear(in_dim, hidden, factorized=factorized)
        self.hidden_layers = nn.ModuleList(_linear(hidden, hidden, factorized=factorized) for _ in range(layers))
        self.out_layer = _linear(hidden, out_dim, factorized=factorized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.in_layer(x))
        u = torch.tanh(self.u_layer(x))
        v = torch.tanh(self.v_layer(x))
        for layer in self.hidden_layers:
            gate = torch.sigmoid(layer(h))
            h = (1.0 - gate) * u + gate * v
        return self.out_layer(h)


class PirateBlock(nn.Module):
    def __init__(self, hidden: int, *, factorized: bool) -> None:
        super().__init__()
        self.f_layer = _linear(hidden, hidden, factorized=factorized)
        self.g_layer = _linear(hidden, hidden, factorized=factorized)
        self.h_layer = _linear(hidden, hidden, factorized=factorized)
        self.raw_alpha = nn.Parameter(torch.tensor(-6.0))

    def forward(self, x: torch.Tensor, u_gate: torch.Tensor, v_gate: torch.Tensor) -> torch.Tensor:
        f = torch.tanh(self.f_layer(x))
        z1 = f * u_gate + (1.0 - f) * v_gate
        g = torch.tanh(self.g_layer(z1))
        z2 = g * u_gate + (1.0 - g) * v_gate
        h = torch.tanh(self.h_layer(z2))
        alpha = torch.sigmoid(self.raw_alpha)
        return alpha * h + (1.0 - alpha) * x


class PirateNet(nn.Module):
    """PirateNet-style gated residual MLP for PINNs.

    The adaptive skip starts nearly at the identity map and gradually lets
    residual blocks contribute, mirroring the moving-interface PINN paper's
    use of PirateNet for dynamic fronts.
    """

    def __init__(self, in_dim: int, hidden: int, blocks: int, out_dim: int, *, factorized: bool = False) -> None:
        super().__init__()
        self.in_layer = _linear(in_dim, hidden, factorized=factorized)
        self.u_layer = _linear(in_dim, hidden, factorized=factorized)
        self.v_layer = _linear(in_dim, hidden, factorized=factorized)
        self.blocks = nn.ModuleList(PirateBlock(hidden, factorized=factorized) for _ in range(blocks))
        self.out_layer = _linear(hidden, out_dim, factorized=factorized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.in_layer(x))
        u_gate = torch.tanh(self.u_layer(x))
        v_gate = torch.tanh(self.v_layer(x))
        for block in self.blocks:
            h = block(h, u_gate, v_gate)
        return self.out_layer(h)


def _effective_l1(layer: nn.Module) -> torch.Tensor:
    if isinstance(layer, FactorizedLinear):
        weight = torch.exp(layer.log_scale).view(-1, 1) * layer.weight_v
    else:
        weight = layer.weight
    return weight.abs().mean()


class NIFPirateNet(nn.Module):
    """Last-layer Neural Implicit Flow head with a PirateNet ShapeNet.

    ShapeNet sees only spatial coordinates/features and returns a low-rank
    spatial basis. ParameterNet sees time and PDE parameters and returns the
    last-layer coefficients plus a bias, matching the efficient NIF variant
    that parameterizes only the decoder layer instead of every ShapeNet weight.
    """

    def __init__(
        self,
        spatial_dim: int,
        parameter_dim: int,
        hidden: int,
        blocks: int,
        out_dim: int,
        rank: int,
        *,
        factorized: bool = False,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        parameter_hidden = max(16, hidden // 2, self.rank)
        parameter_layers = max(1, blocks // 2)
        self.shape_net = PirateNet(spatial_dim, hidden, blocks, self.rank, factorized=factorized)
        self.parameter_net = GatedMLP(
            parameter_dim,
            parameter_hidden,
            parameter_layers,
            self.rank * out_dim + out_dim,
            factorized=factorized,
        )

    def forward(self, spatial_features: torch.Tensor, parameter_features: torch.Tensor) -> torch.Tensor:
        basis = torch.tanh(self.shape_net(spatial_features))
        coefficients_and_bias = self.parameter_net(parameter_features)
        coefficients = coefficients_and_bias[:, : self.rank] / math.sqrt(float(self.rank))
        bias = coefficients_and_bias[:, self.rank :]
        return torch.sum(basis * coefficients, dim=-1, keepdim=True) + bias

    def sparse_l1(self) -> torch.Tensor:
        return 0.5 * (_effective_l1(self.shape_net.out_layer) + _effective_l1(self.parameter_net.out_layer))


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
        self.architecture = model.architecture
        self.use_nif_head = model.architecture == "nif_pirate"
        self.use_geo_features = bool(model.use_geo_features)
        self.spatial_fourier_only = bool(model.spatial_fourier_only)
        self.use_seed_front_features = bool(model.use_seed_front_features)
        self.use_traveling_wave_features = bool(model.use_traveling_wave_features)
        self.hard_initial_condition = bool(model.hard_initial_condition)
        self.initial_envelope_tau = float(model.initial_envelope_tau)
        self.use_kpp_front_envelope = bool(model.use_kpp_front_envelope)
        self.front_envelope_margin = float(model.front_envelope_margin)
        self.front_envelope_width = float(model.front_envelope_width)
        self.seed_sigma = float(seed.sigma)
        self.seed_amplitude = float(seed.amplitude)
        self.reference_diffusion = float(pde.diffusion)
        self.reference_reaction = float(pde.reaction)
        self.reference_front_speed = float(2.0 * math.sqrt(max(pde.diffusion, 1.0e-12) * max(pde.reaction, 1.0e-12)))
        self.register_buffer("seed_center", torch.tensor([seed.center_x, seed.center_y], dtype=torch.float32))
        fourier_dim = 2 if (self.spatial_fourier_only or self.use_nif_head) else 3
        self.features = FourierFeatures(fourier_dim, model.fourier_features, model.fourier_sigma)
        geo_dim = 8 if self.use_geo_features else 0
        seed_front_dim = 4 if self.use_seed_front_features else 0
        traveling_wave_dim = 5 if (self.use_traveling_wave_features and not self.use_nif_head) else 0
        network_in_dim = 2 * model.fourier_features + 3 + geo_dim + seed_front_dim + traveling_wave_dim
        nif_spatial_dim = 2 * model.fourier_features + 2 + geo_dim + seed_front_dim
        nif_parameter_dim = 4
        if model.architecture == "gated_mlp":
            self.mlp = GatedMLP(
                network_in_dim,
                model.hidden,
                model.layers,
                1,
                factorized=model.use_random_weight_factorization,
            )
        elif model.architecture == "pirate":
            self.mlp = PirateNet(
                network_in_dim,
                model.hidden,
                model.layers,
                1,
                factorized=model.use_random_weight_factorization,
            )
        elif model.architecture == "nif_pirate":
            self.mlp = NIFPirateNet(
                nif_spatial_dim,
                nif_parameter_dim,
                model.hidden,
                model.layers,
                1,
                model.nif_rank,
                factorized=model.use_random_weight_factorization,
            )
        else:
            raise ValueError(f"Unknown ModelConfig.architecture={model.architecture!r}")
        self.source = SourceHead(domain, seed)
        self.pde = PDEParameters(pde, model)

    def forward(self, xy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.use_nif_head:
            neural_field = torch.sigmoid(self.mlp(self._nif_spatial_features(xy), self._nif_parameter_features(t)))
        else:
            neural_field = torch.sigmoid(self.mlp(self._joint_features(xy, t)))
        if self.use_kpp_front_envelope:
            neural_field = self._front_envelope(xy, t) * neural_field
        if self.hard_initial_condition:
            initial_field = self._known_initial_profile(xy).clamp(0.0, 1.0)
            blend = torch.exp(-t / (self.initial_envelope_tau * self.domain.t_end + 1.0e-8))
            return blend * initial_field + (1.0 - blend) * neural_field
        if not self.use_source_envelope:
            return neural_field
        source_field = self.source.profile(xy).clamp(0.0, 1.0)
        # A hard initial-source envelope: exactly source-dominated at t=0, but
        # negligible in the late observation window. This gives source parameters
        # gradients through the PDE residual instead of treating origin extraction
        # as a post-hoc argmax.
        blend = torch.exp(-t / (0.08 * self.domain.t_end + 1.0e-8))
        return blend * source_field + (1.0 - blend) * neural_field

    def _joint_features(self, xy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xyt = torch.cat([xy / self.domain.box, t / self.domain.t_end], dim=-1)
        spectral_input = xyt[:, :2] if self.spatial_fourier_only else xyt
        pieces = [xyt, self.features(spectral_input)]
        if self.use_geo_features:
            pieces.append(square_geo_features(xy, self.domain.box))
        if self.use_seed_front_features:
            pieces.append(
                seed_front_features(
                    xy,
                    t,
                    self.seed_center,
                    self.seed_sigma,
                    self.reference_diffusion,
                    self.reference_front_speed,
                    self.domain.box,
                )
            )
        if self.use_traveling_wave_features:
            pieces.append(
                traveling_wave_features(
                    xy,
                    t,
                    self.seed_center,
                    self.seed_sigma,
                    self.pde.diffusion(),
                    self.pde.reaction(),
                    self.domain.box,
                )
            )
        return torch.cat(pieces, dim=-1)

    def _nif_spatial_features(self, xy: torch.Tensor) -> torch.Tensor:
        xy_scaled = xy / self.domain.box
        pieces = [xy_scaled, self.features(xy_scaled)]
        if self.use_geo_features:
            pieces.append(square_geo_features(xy, self.domain.box))
        if self.use_seed_front_features:
            pieces.append(seed_spatial_features(xy, self.seed_center, self.seed_sigma, self.domain.box))
        return torch.cat(pieces, dim=-1)

    def _nif_parameter_features(self, t: torch.Tensor) -> torch.Tensor:
        t_scaled = t / self.domain.t_end
        diffusion = self.pde.diffusion().clamp_min(1.0e-12)
        reaction = self.pde.reaction().clamp_min(1.0e-12)
        diffusion_scale = diffusion / max(self.reference_diffusion, 1.0e-12)
        reaction_scale = reaction / max(self.reference_reaction, 1.0e-12)
        front_speed = 2.0 * torch.sqrt(diffusion * reaction)
        front_radius = (3.0 * self.seed_sigma + front_speed * t) / self.domain.box
        return torch.cat(
            [
                t_scaled,
                torch.ones_like(t) * diffusion_scale,
                torch.ones_like(t) * reaction_scale,
                front_radius,
            ],
            dim=-1,
        )

    def _known_initial_profile(self, xy: torch.Tensor) -> torch.Tensor:
        center = self.seed_center.to(device=xy.device, dtype=xy.dtype).view(1, 2)
        sigma = torch.as_tensor(self.seed_sigma, dtype=xy.dtype, device=xy.device)
        dist2 = ((xy - center) ** 2).sum(dim=-1, keepdim=True)
        return self.seed_amplitude * torch.exp(-dist2 / (2.0 * sigma**2))

    def _front_envelope(self, xy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        center = self.seed_center.to(device=xy.device, dtype=xy.dtype).view(1, 2)
        radius = torch.linalg.norm(xy - center, dim=-1, keepdim=True)
        speed = torch.as_tensor(self.reference_front_speed, dtype=xy.dtype, device=xy.device)
        base_radius = 3.0 * self.seed_sigma + self.front_envelope_margin
        support_radius = base_radius + speed * t
        width = max(self.front_envelope_width, 1.0e-4)
        return torch.sigmoid((support_radius - radius) / width)

    def sparse_last_layer_l1(self) -> torch.Tensor:
        if hasattr(self.mlp, "sparse_l1"):
            return self.mlp.sparse_l1()
        out_layer = self.mlp.out_layer
        return _effective_l1(out_layer)

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
