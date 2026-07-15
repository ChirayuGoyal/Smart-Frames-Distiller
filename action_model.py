"""
Action recognition backend for frame selection.

Primary: torchvision video model (R3D-18 Kinetics-400) — practical stand-in for
X3D-S cited in RESEARCH.md (Feichtenhofer 2020).

Fallback: motion-energy feature vector when torch/torchvision unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Project-local weights cache — avoids re-downloading on every run.
# First run: downloads via torchvision and saves here.
# Subsequent runs: loads directly from this file (no network).
_MODEL_CACHE_PATH = Path(__file__).resolve().parent / "models" / "r3d18_weights.pt"

try:
    import torch
    import torch.nn.functional as F
    from torchvision.models.video import R3D_18_Weights, r3d_18

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class ActionPrediction:
    frame_index: int
    class_id: int
    confidence: float
    top_label: str
    logits: np.ndarray | None = None


class ActionModelBackend:
    def predict_clip(self, frames: list[np.ndarray], center_index: int) -> ActionPrediction:
        raise NotImplementedError

    def predict_batch(
        self, clips_with_indices: list[tuple[int, list[np.ndarray]]]
    ) -> list[ActionPrediction]:
        """Default: serial loop.  GPU backends override this for batched inference."""
        return [self.predict_clip(clip, idx) for idx, clip in clips_with_indices]


def resolve_device(device: str | None = None) -> str:
    if device is None or device == "auto":
        if _TORCH_AVAILABLE:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"
    return device.lower()


class TorchvisionActionModel(ActionModelBackend):
    """R3D-18 on 16-frame clips, per RESEARCH.md clip_len=16."""

    def __init__(
        self,
        clip_len: int = 16,
        device: str | None = None,
        model_path: str | None = None,
        cache_dir: str | None = None,
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch/torchvision required for TorchvisionActionModel")

        self.clip_len = clip_len
        self.device   = resolve_device(device)

        weights = R3D_18_Weights.DEFAULT
        self.categories = weights.meta["categories"]

        local_cache = Path(model_path) if model_path else _MODEL_CACHE_PATH

        if local_cache.is_file():
            # Fast path: load from project-local cache — no network access
            log.info("loading R3D-18 from local cache: %s", local_cache)
            self.model = r3d_18()
            try:
                state = torch.load(local_cache, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(local_cache, map_location="cpu")
            self.model.load_state_dict(state)
            self.model = self.model.to(self.device).eval()
        else:
            # First run: download via torchvision then save locally for next time
            if cache_dir:
                torch.hub.set_dir(cache_dir)
            log.info("downloading R3D-18 weights (will be cached at %s)", local_cache)
            self.model = r3d_18(weights=weights).to(self.device).eval()
            try:
                local_cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.cpu().state_dict(), local_cache)
                self.model = self.model.to(self.device)
                log.info("R3D-18 weights cached → %s", local_cache)
            except Exception as exc:
                log.warning("could not save model cache: %s", exc)

        # Cache on CPU — moved to device once per batch, not once per frame
        self._mean_cpu = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
        self._std_cpu  = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)

    def _clip_to_tensor(self, frames: list[np.ndarray]) -> torch.Tensor:
        """
        Convert a list of BGR frames → (1, C, T, H, W) float32 normalized on CPU.
        Keeps the result on CPU so callers can cat many clips before one .to(device).
        """
        # Stack + BGR→RGB in one shot; .copy() makes the array contiguous
        arr = np.stack([f[:, :, ::-1] for f in frames], axis=0).copy()  # (T,H,W,C)
        t   = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)  # (T,C,H,W)
        # Resize all T frames at once — was a per-frame Python loop before
        t   = F.interpolate(t, size=(112, 112), mode="bilinear", align_corners=False)
        t.sub_(self._mean_cpu).div_(self._std_cpu)
        return t.permute(1, 0, 2, 3).unsqueeze(0)  # (1,C,T,H,W)

    def predict_clip(self, frames: list[np.ndarray], center_index: int) -> ActionPrediction:
        tensor = self._clip_to_tensor(frames).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = F.softmax(logits, dim=1)[0]
        conf, cls = torch.max(probs, dim=0)
        ci = int(cls)
        return ActionPrediction(
            frame_index=center_index, class_id=ci, confidence=float(conf),
            top_label=self.categories[ci], logits=probs.cpu().numpy(),
        )

    def predict_batch(
        self, clips_with_indices: list[tuple[int, list[np.ndarray]]]
    ) -> list[ActionPrediction]:
        """
        Run all clips in a single GPU forward pass.
        All clips are normalised on CPU then transferred in one .to(device) call.
        """
        batch = torch.cat(
            [self._clip_to_tensor(clip) for _, clip in clips_with_indices], dim=0
        ).to(self.device)                          # (B, C, T, H, W)  — one H2D transfer

        with torch.no_grad():
            logits = self.model(batch)             # (B, 400)
            probs  = F.softmax(logits, dim=1)      # (B, 400)

        results = []
        for i, (center_idx, _) in enumerate(clips_with_indices):
            conf, cls = torch.max(probs[i], dim=0)
            ci = int(cls)
            results.append(ActionPrediction(
                frame_index=center_idx, class_id=ci, confidence=float(conf),
                top_label=self.categories[ci], logits=probs[i].cpu().numpy(),
            ))
        return results


class MotionEnergyActionModel(ActionModelBackend):
    """
    CPU fallback: quantize motion energy + edge density into pseudo action classes.
    Not Kinetics-calibrated but detects regime changes for validation without GPU.
    """

    LABELS = ["static_low", "static_high", "motion_low", "motion_high"]

    def predict_clip(self, frames: list[np.ndarray], center_index: int) -> ActionPrediction:
        if len(frames) < 2:
            return ActionPrediction(center_index, 0, 1.0, self.LABELS[0])

        energies = []
        edges    = []
        # Compute grayscale once per frame, reuse across consecutive pairs
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        for i in range(1, len(grays)):
            diff = cv2.absdiff(grays[i - 1], grays[i])
            energies.append(float(np.mean(diff)))
            edges.append(float(np.mean(cv2.Canny(grays[i], 50, 150))))

        energy   = float(np.mean(energies))
        edge     = float(np.mean(edges))
        e_bin    = 1 if energy > 8.0 else 0
        s_bin    = 1 if edge   > 12.0 else 0
        class_id = e_bin * 2 + s_bin
        confidence = min(1.0, 0.5 + energy / 40.0)
        logits   = np.zeros(len(self.LABELS), dtype=np.float32)
        logits[class_id] = confidence
        logits  /= logits.sum() if logits.sum() > 0 else 1.0
        return ActionPrediction(
            frame_index=center_index, class_id=class_id, confidence=confidence,
            top_label=self.LABELS[class_id], logits=logits,
        )


import cv2  # noqa: E402  (late import — keeps optional at module level)


class EnsembleActionModel(ActionModelBackend):
    """
    Combines TorchvisionActionModel (R3D-18) and MotionEnergyActionModel.

    Trigger logic (OR): a frame change is flagged when EITHER sub-model
    detects a meaningful change.

    Implementation:
      combined_class_id  = r3d_class * 4 + motion_class
        → label_changed fires when either sub-class changes
      combined_confidence = 0.7 × r3d_conf + 0.3 × motion_conf
        → conf_jump is weighted toward R3D-18 but motion energy adds signal
      logits = R3D-18 logits (used for cosine correlation scores)
    """

    _MOTION_N = len(MotionEnergyActionModel.LABELS)  # 4
    _W_TORCH  = 0.7
    _W_MOTION = 0.3

    def __init__(
        self,
        clip_len: int = 16,
        device: str | None = None,
        model_path: str | None = None,
        cache_dir: str | None = None,
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch/torchvision required for EnsembleActionModel")
        self.torch_model  = TorchvisionActionModel(
            clip_len=clip_len, device=device,
            model_path=model_path, cache_dir=cache_dir,
        )
        self.motion_model = MotionEnergyActionModel()

    def _combine(self, tp: ActionPrediction, mp: ActionPrediction) -> ActionPrediction:
        return ActionPrediction(
            frame_index=tp.frame_index,
            class_id=tp.class_id * self._MOTION_N + mp.class_id,
            confidence=self._W_TORCH * tp.confidence + self._W_MOTION * mp.confidence,
            top_label=f"{tp.top_label}|{mp.top_label}",
            logits=tp.logits,   # R3D-18 logits for cosine correlation
        )

    def predict_clip(self, frames: list[np.ndarray], center_index: int) -> ActionPrediction:
        tp = self.torch_model.predict_clip(frames, center_index)
        mp = self.motion_model.predict_clip(frames, center_index)
        return self._combine(tp, mp)

    def predict_batch(
        self, clips_with_indices: list[tuple[int, list[np.ndarray]]]
    ) -> list[ActionPrediction]:
        torch_preds  = self.torch_model.predict_batch(clips_with_indices)   # batched GPU
        motion_preds = self.motion_model.predict_batch(clips_with_indices)  # serial CPU
        return [self._combine(tp, mp) for tp, mp in zip(torch_preds, motion_preds)]


def create_action_model(
    clip_len: int = 16,
    prefer_torch: bool = True,
    device: str | None = None,
    ensemble: bool = False,
    model_path: str | None = None,
    cache_dir: str | None = None,
) -> ActionModelBackend:
    resolved = resolve_device(device)
    if prefer_torch and _TORCH_AVAILABLE:
        try:
            if ensemble:
                return EnsembleActionModel(
                    clip_len=clip_len, device=resolved,
                    model_path=model_path, cache_dir=cache_dir,
                )
            return TorchvisionActionModel(
                clip_len=clip_len, device=resolved,
                model_path=model_path, cache_dir=cache_dir,
            )
        except Exception:
            pass
    return MotionEnergyActionModel()
