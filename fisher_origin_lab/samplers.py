from __future__ import annotations

import torch

from .models import OriginPINN
from .losses import front_indicator_weights, pde_residual_terms


class SobolCollocation:
    def __init__(self, box: float, t_end: float, device: torch.device, seed: int, mask_kind: str = "box") -> None:
        self.box = box
        self.t_end = t_end
        self.device = device
        self.mask_kind = mask_kind
        self.engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)
        self.anchors: torch.Tensor | None = None

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        random_part = self._draw_valid(n)
        if self.anchors is None or len(self.anchors) == 0:
            points = random_part
        else:
            anchor_n = min(len(self.anchors), max(1, n // 4))
            idx = torch.randint(0, len(self.anchors), (anchor_n,), device=self.device)
            points = torch.cat([random_part[: n - anchor_n], self.anchors[idx]], dim=0)
        return points[:, :2], points[:, 2:3]

    def refresh(
        self,
        model: OriginPINN,
        candidate_n: int,
        keep: int,
        chunk: int = 1024,
        front_alpha: float = 0.0,
        front_gradient: float = 0.0,
    ) -> None:
        candidates = self._draw_valid(candidate_n)
        scores = []
        for start in range(0, candidate_n, chunk):
            part = candidates[start : start + chunk]
            with torch.enable_grad():
                residual, u, u_xy, _, _ = pde_residual_terms(model, part[:, :2], part[:, 2:3])
                front = front_indicator_weights(u, u_xy, front_alpha, front_gradient)
                score = residual.detach().abs() * front.detach()
            scores.append(score.flatten())
        score = torch.cat(scores)
        topk = min(keep, len(score))
        idx = torch.topk(score, k=topk).indices
        self.anchors = candidates[idx].detach()

    def _draw_valid(self, n: int) -> torch.Tensor:
        if self.mask_kind == "box":
            points = self.engine.draw(n).to(self.device)
            points[:, :2] *= self.box
            points[:, 2:3] *= self.t_end
            return points

        chunks = []
        remaining = n
        while remaining > 0:
            candidate = self.engine.draw(max(remaining * 2, 32)).to(self.device)
            candidate[:, :2] *= self.box
            candidate[:, 2:3] *= self.t_end
            valid = self._valid_mask(candidate[:, :2])
            chosen = candidate[valid][:remaining]
            if len(chosen) == 0:
                raise ValueError(f"mask_kind={self.mask_kind!r} produced no valid collocation points.")
            chunks.append(chosen)
            remaining -= len(chosen)
        return torch.cat(chunks, dim=0)

    def _valid_mask(self, xy: torch.Tensor) -> torch.Tensor:
        if self.mask_kind == "ellipse":
            scaled = xy / self.box - 0.5
            return (scaled[:, 0] / 0.48).pow(2) + (scaled[:, 1] / 0.42).pow(2) <= 1.0
        if self.mask_kind != "box":
            raise ValueError(f"Unknown mask_kind={self.mask_kind!r}; expected 'box' or 'ellipse'.")
        return torch.ones(len(xy), dtype=torch.bool, device=xy.device)
