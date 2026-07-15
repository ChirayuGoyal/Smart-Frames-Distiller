    # Action-Aware Methods

> **Status:** Implemented + cross-validation harness (2026-06-10)

## Implementation

See `action_model.py`, `selector.py`, `main.py`, `validate.py`.

- **Model:** R3D-18 Kinetics-400 (16-frame clips) with motion-energy CPU fallback
- **CLI:** `python main.py video.mp4 --write-video`
- **Validate:** `python validate.py` (synthetic video + metrics)  
> **Search queries:** `VideoMAE action recognition video summarization`, `InternVideo long-range action understanding`, `TimeSformer event detection`

## Summary

Action recognition models produce **temporal embeddings or class logits** that shift when actions change. Keep frames where action label or confidence changes significantly; skip frames within stable action segments.

## Proposed POC

```
# Sample every k frames for efficiency
logits_t = action_model(clip around t)  # VideoMAE, InternVideo2, X3D
keep(t) if argmax(logits_t) != argmax(logits_{t-k}) 
         OR |max_conf_t - max_conf_{t-k}| > delta
```

## Related Work

- **Tang et al. (2023, ACM TOMM):** CNN deep features + Temporal Segment Density Peaks Clustering for keyframes — action classification downstream benefits
- **VIRAT / Sports-1M:** Action-labeled data for evaluating action retention post-compression
- **Video-LLM selectors (AKS, KeyVideoLLM):** Query-aware but not action-class-aware — complementary

## Model Options

| Model | Input | Strength | Latency |
|-------|-------|----------|---------|
| X3D-S | 16-frame clip | Efficient 3D CNN | ~20 ms GPU |
| VideoMAE-V2 | 16–32 frames | Self-supervised, fine-tunable | ~50 ms GPU |
| InternVideo2 | Variable | SOTA action understanding | ~100+ ms GPU |
| TimeSformer | Long clips | Long-range attention | High memory |

## Failure Modes

- **Compound actions** (walking while talking): Label stable, semantics shift
- **Sampling gap:** Miss brief actions between sampled frames
- **Domain gap:** Model trained on Kinetics, fails on surveillance micro-actions

## Literature Gap

**Few papers** explicitly use action classifier confidence shifts for **compression frame dropping**. Primarily summarization/QoE literature. **Novel research path.**

## POC Mapping

| POC | Action method |
|-----|--------------|
| A | InternVideo2 + X3D ensemble |
| B | X3D-S on 16-frame windows |
| C | Skip (too slow) |
| D | Skip |
| E | Batch VideoMAE on cloud |

## Sources

1. Tang, H. et al. (2023). *Deep Unsupervised Key Frame Extraction for Efficient Video Classification.* ACM TOMM. doi:10.1145/3571735
2. Feichtenhofer, C. (2020). *X3D: Expanding Architectures for Efficient Video Recognition.* CVPR.
