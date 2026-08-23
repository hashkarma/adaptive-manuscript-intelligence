# Difficult Adaptive Manuscript

## Demo purpose
Case 2 proves judgement.

The manuscript achieves strong segmentation, but Stage-5 recognition
readiness and Stage-6 semantic trust remain insufficient. The adaptive
controller therefore invokes an independent Qwen Provider-C retry. Because
the retry does not improve the existing readiness metric, the system routes
to scholar review rather than manufacturing a trusted translation.

## Canonical metrics
- S: 0.8854
- H(A+B): 0.3371
- T: 0.0
- H(A+C): 0.3297
- H(B+C): 0.3298
- Selected retry pair: B+C
- Selected retry H: 0.3298
- Delta H: -0.0073
- Provider C improved baseline H: False

## Final adaptive outcome
- Final status: review_required
- Final decision: scholar_review_after_machine_exhaustion
- Next action: route_to_scholar_review
- Machine retry exhausted: True
- Scholar review required: True

## Scientific guardrails
- H is an HTR readiness signal, not accuracy.
- T is a semantic trust signal, not accuracy.
- Ground-truth CER/WER are unavailable.
- Provider C not improving H does not prove linguistic inferiority.
- Machine exhaustion is stated only under the currently declared executable
  capability set.
