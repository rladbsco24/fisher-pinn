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

    def sample(
        self,
        n: int,
        *,
        t_low: float | None = None,
        t_high: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        random_part = self._draw_valid(n, t_low=t_low, t_high=t_high)
        if self.anchors is None or len(self.anchors) == 0:
            points = random_part
        else:
            anchors = self.anchors
            if t_low is not None or t_high is not None:
                low = 0.0 if t_low is None else float(t_low)
                high = self.t_end if t_high is None else float(t_high)
                keep_mask = (anchors[:, 2] >= low) & (anchors[:, 2] <= high)
                anchors = anchors[keep_mask]
            if len(anchors) == 0:
                points = random_part
                return points[:, :2], points[:, 2:3]
            anchor_n = min(len(anchors), max(1, n // 4))
            idx = torch.randint(0, len(anchors), (anchor_n,), device=self.device)
            points = torch.cat([random_part[: n - anchor_n], anchors[idx]], dim=0)
        return points[:, :2], points[:, 2:3]

    def refresh(
        self,
        model: OriginPINN,
        candidate_n: int,
        keep: int,
        chunk: int = 1024,
        front_alpha: float = 0.0,
        front_gradient: float = 0.0,
        residual_weight: float = 1.0,
        gradient_weight: float = 0.25,
        activity_weight: float = 0.25,
        t_low: float | None = None,
        t_high: float | None = None,
    ) -> None:
        candidates = self._draw_valid(candidate_n, t_low=t_low, t_high=t_high)
        scores = []
        for start in range(0, candidate_n, chunk):
            part = candidates[start : start + chunk]
            with torch.enable_grad():
                residual, u, u_xy, _, _ = pde_residual_terms(model, part[:, :2], part[:, 2:3])
                front = front_indicator_weights(u, u_xy, front_alpha, front_gradient).detach()
                residual_score = _normalize_score(residual.detach().abs()) * front
                gradient_score = _normalize_score(torch.linalg.norm(u_xy.detach(), dim=-1, keepdim=True))
                activity = u.detach().clamp(0.0, 1.0) * (1.0 - u.detach().clamp(0.0, 1.0))
                activity_score = _normalize_score(activity)
                score = (
                    float(residual_weight) * residual_score
                    + float(gradient_weight) * gradient_score
                    + float(activity_weight) * activity_score
                )
            scores.append(score.flatten())
        score = torch.cat(scores)
        topk = min(keep, len(score))
        idx = torch.topk(score, k=topk).indices
        self.anchors = candidates[idx].detach()

    def _draw_valid(
        self,
        n: int,
        *,
        t_low: float | None = None,
        t_high: float | None = None,
    ) -> torch.Tensor:
        low = 0.0 if t_low is None else max(0.0, min(float(t_low), self.t_end))
        high = self.t_end if t_high is None else max(0.0, min(float(t_high), self.t_end))
        if high < low:
            low, high = high, low
        span = max(high - low, 1.0e-8)
        if self.mask_kind == "box":
            points = self.engine.draw(n).to(self.device)
            points[:, :2] *= self.box
            points[:, 2:3] = low + span * points[:, 2:3]
            return points

        chunks = []
        remaining = n
        while remaining > 0:
            candidate = self.engine.draw(max(remaining * 2, 32)).to(self.device)
            candidate[:, :2] *= self.box
            candidate[:, 2:3] = low + span * candidate[:, 2:3]
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


def _normalize_score(score: torch.Tensor) -> torch.Tensor:
    return score / score.mean().clamp_min(1.0e-8)
