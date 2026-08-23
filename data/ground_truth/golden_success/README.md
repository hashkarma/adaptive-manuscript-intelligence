# Golden Success — Professor Ground Truth

This directory contains the canonical structured Ground Truth derivative for
the Golden manuscript benchmark.

## Provenance

The Sanskrit reference text is professor supplied from Page 1 of the source
ODT. The source document is preserved outside the Git-tracked dataset and is
identified by SHA-256 in `provenance.json`.

The mapping of the professor text onto the 11 Stage-4 physical manuscript
lines was created by project visual review. The physical line breaks must not
be described as professor supplied.

## Metrics

Stage-5 transcription may be evaluated using CER/WER against this reference.

- Strict CER retains whitespace and punctuation.
- WER uses whitespace tokenization and is spacing-sensitive.
- Content CER is a secondary diagnostic that removes whitespace/separators and
  punctuation only.

H remains the Stage-5 readiness/evidence signal. T remains the Stage-6 trust
signal. Neither is replaced by CER/WER.

The professor source does not provide validated English or Hindi translation
Ground Truth. Stage-6 translations remain AI candidates until separately
scholar validated.

## Publication

Confirm redistribution permission before publishing the professor-supplied
transcription or original ODT outside the research/project context.
