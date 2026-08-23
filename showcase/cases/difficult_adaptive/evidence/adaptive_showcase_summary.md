# Adaptive Manuscript Showcase — Canonical Evidence Summary

## Pipeline outcome

- Stage 4 segmentation: **S = 0.8854**, 11 lines.
- Stage 5 baseline A+B: **H = 0.3371**.
- A/B mean content agreement: **0.2823**.
- Stage 6 reconstruction: **0 reconstructed, 11 abstained**.
- Stage 6 semantic trust: **T = 0.0**.
- Provider C (Qwen) retry:
  - A+C H = **0.3297**
  - B+C H = **0.3298**
  - selected pair = **B+C**
  - delta H = **-0.0073**
  - improved baseline = **False**
- Final orchestrator decision: **scholar_review_after_machine_exhaustion**.
- Final next action: **route_to_scholar_review**.

## Interpretation

The adaptive controller detected weak recognition evidence despite strong segmentation, allowed Stage 6 semantic recovery, selected a materially different third HTR provider after T remained unresolved, measured the retry using the existing pairwise H(p) definition, rejected the retry because it did not improve H(p), and only then authorized scholar review under the currently declared machine capabilities.

## Safety / research claims

H(p) and T(p) are readiness/trust signals, not recognition accuracy or calibrated probabilities. CER/WER remain unavailable because scholar-verified ground truth is not available. The Provider-C delta-H result does not by itself establish whether Qwen's transcription is linguistically better or worse.
