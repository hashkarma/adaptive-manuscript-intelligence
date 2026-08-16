from __future__ import annotations

import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Allow unsupported MPS operations to fall back to CPU when PyTorch supports it.
# This must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from PIL import Image


# ---------------------------------------------------------------------------
# Provider model / processor IDs
# ---------------------------------------------------------------------------

# Provider A
DEFAULT_TROCR_IAST_MODEL_ID = "Piyush3142/trocr-sanskrit-ocr"

# Provider B
DEFAULT_TROCR_VEDIC_MODEL_ID = "yzk/trocr-large-printed-vedic"

# IMPORTANT:
# The Vedic fine-tuned repository does not expose processor assets in the
# format expected by transformers==4.42.1. Its processor/tokenizer lineage
# comes from the Microsoft TrOCR large printed base, so Provider B deliberately
# loads:
#
#   processor -> microsoft/trocr-large-printed
#   model     -> yzk/trocr-large-printed-vedic
#
DEFAULT_TROCR_VEDIC_PROCESSOR_ID = "microsoft/trocr-large-printed"


# ---------------------------------------------------------------------------
# Provider contracts
# ---------------------------------------------------------------------------

@dataclass
class ProviderHypothesis:
    rank: int
    raw_text: str
    relative_score: Optional[float]
    sequence_score: Optional[float]


@dataclass
class ProviderResult:
    hypotheses: List[ProviderHypothesis]
    runtime_ms: float
    device_used: str
    metadata: Dict[str, Any]


class HTRProvider(ABC):
    """
    Inference-only contract for Research Stage 5 HTR providers.

    A provider must:
      1. accept a single line image,
      2. emit N-best raw text hypotheses,
      3. expose runtime/provider metadata.

    A provider must NOT decide:
      - H(p)
      - CER/WER
      - scholarly correctness
      - orchestrator routing
    """

    provider_id: str
    output_script: str

    @property
    @abstractmethod
    def model_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def processor_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def device_info(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def recognize(
        self,
        image: Image.Image,
        *,
        num_beams: int,
        n_best: int,
        max_output_length: int,
    ) -> ProviderResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _resolve_device(requested: str = "auto") -> Tuple[str, Dict[str, Any]]:
    import torch

    requested = requested.lower().strip()

    if requested not in {"auto", "mps", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, mps, cuda, cpu")

    mps_available = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    cuda_available = bool(torch.cuda.is_available())

    if requested == "auto":
        if mps_available:
            selected = "mps"
        elif cuda_available:
            selected = "cuda"
        else:
            selected = "cpu"

    elif requested == "mps":
        if not mps_available:
            raise RuntimeError(
                "MPS was requested but is unavailable. "
                "Use native arm64 Python with an MPS-enabled PyTorch build."
            )
        selected = "mps"

    elif requested == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA was requested but is unavailable.")
        selected = "cuda"

    else:
        selected = "cpu"

    return selected, {
        "requested_device": requested,
        "selected_device": selected,
        "torch_version": torch.__version__,
        "mps_available": mps_available,
        "cuda_available": cuda_available,
    }


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []

    maximum = max(float(v) for v in values)
    exps = [
        math.exp(float(v) - maximum)
        for v in values
    ]

    total = sum(exps)

    if total <= 0:
        return [0.0 for _ in values]

    return [
        value / total
        for value in exps
    ]


# ---------------------------------------------------------------------------
# Shared TrOCR provider implementation
# ---------------------------------------------------------------------------

class _TrOCRProviderBase(HTRProvider):
    """
    Shared inference implementation for Hugging Face TrOCR providers.

    processor_id and model_id are intentionally separate.

    This is critical for Provider B:

        processor_id = microsoft/trocr-large-printed
        model_id     = yzk/trocr-large-printed-vedic
    """

    def __init__(
        self,
        model_id: str,
        *,
        processor_id: Optional[str] = None,
        device: str = "auto",
        allow_cpu_fallback: bool = True,
    ) -> None:

        self._model_id = model_id
        self._processor_id = processor_id or model_id

        self._device, self._device_info = _resolve_device(
            device
        )

        self.allow_cpu_fallback = allow_cpu_fallback

        try:
            import torch
            from transformers import (
                TrOCRProcessor,
                VisionEncoderDecoderModel,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Stage 5 dependencies are missing. Install: "
                "torch torchvision transformers safetensors "
                "pillow indic-transliteration"
            ) from exc

        self._torch = torch
        self._TrOCRProcessor = TrOCRProcessor
        self._VisionEncoderDecoderModel = (
            VisionEncoderDecoderModel
        )

        self.processor = None
        self.model = None

    # ------------------------------------------------------------------
    # Public provider properties
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def processor_id(self) -> str:
        return self._processor_id

    @property
    def device(self) -> str:
        return self._device

    @property
    def device_info(self) -> Dict[str, Any]:
        info = dict(self._device_info)

        info.update(
            {
                "provider_id": self.provider_id,
                "model_id": self._model_id,
                "processor_id": self._processor_id,
                "output_script": self.output_script,
            }
        )

        return info

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _validate_materialized_model(self) -> None:
        """
        Guard against silently using a partially materialized model.

        A meta tensor contains shape information but no real tensor data.
        We never call to_empty() or manually initialize such weights because
        that could silently create an invalid OCR model.
        """

        meta_parameters = [
            name
            for name, parameter
            in self.model.named_parameters()
            if parameter.device.type == "meta"
        ]

        meta_buffers = [
            name
            for name, buffer
            in self.model.named_buffers()
            if buffer.device.type == "meta"
        ]

        if meta_parameters or meta_buffers:
            details: List[str] = []

            if meta_parameters:
                details.append(
                    "meta parameters: "
                    + ", ".join(meta_parameters[:10])
                )

            if meta_buffers:
                details.append(
                    "meta buffers: "
                    + ", ".join(meta_buffers[:10])
                )

            raise RuntimeError(
                "HTR model contains unmaterialized meta tensors; "
                "refusing inference. "
                + " | ".join(details)
            )

    def _validate_tokenizer_model_compatibility(self) -> None:
        """
        Compare tokenizer vocabulary size to decoder vocabulary size where
        both are available.

        This is particularly useful for Provider B because its processor comes
        from the Microsoft base while its weights come from the Vedic fine-tune.
        """

        try:
            tokenizer_vocab_size = len(
                self.processor.tokenizer
            )
        except Exception:
            tokenizer_vocab_size = None

        try:
            decoder_vocab_size = int(
                self.model.config.decoder.vocab_size
            )
        except Exception:
            decoder_vocab_size = None

        if (
            tokenizer_vocab_size is not None
            and decoder_vocab_size is not None
            and tokenizer_vocab_size
            != decoder_vocab_size
        ):
            raise RuntimeError(
                "Tokenizer/model vocabulary mismatch: "
                f"processor vocab={tokenizer_vocab_size}, "
                f"decoder vocab={decoder_vocab_size}"
            )

    def load(self) -> None:
        """
        Load processor and model separately.

        low_cpu_mem_usage=False is intentional. It avoids the lazy/meta-tensor
        loading behavior that caused the earlier Provider-B failure.
        """

        if (
            self.processor is not None
            and self.model is not None
        ):
            return

        # Processor may intentionally come from a different repository
        # than the fine-tuned model.
        self.processor = (
            self._TrOCRProcessor.from_pretrained(
                self._processor_id
            )
        )

        self.model = (
            self._VisionEncoderDecoderModel.from_pretrained(
                self._model_id,
                low_cpu_mem_usage=False,
            )
        )

        # Validate while the model is still on CPU.
        self._validate_materialized_model()
        self._validate_tokenizer_model_compatibility()

        self.model.eval()

        # Only move after validating that all weights/buffers are real.
        self.model.to(
            self._device
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_once(
        self,
        image: Image.Image,
        *,
        num_beams: int,
        n_best: int,
        max_output_length: int,
    ):
        self.load()

        pixel_values = (
            self.processor(
                images=image,
                return_tensors="pt",
            )
            .pixel_values
            .to(self._device)
        )

        with self._torch.inference_mode():
            generated = self.model.generate(
                pixel_values,
                num_beams=num_beams,
                num_return_sequences=n_best,
                max_length=max_output_length,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        return generated

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def recognize(
        self,
        image: Image.Image,
        *,
        num_beams: int = 4,
        n_best: int = 3,
        max_output_length: int = 192,
    ) -> ProviderResult:

        if n_best < 1:
            raise ValueError(
                "n_best must be >= 1"
            )

        if num_beams < n_best:
            raise ValueError(
                "num_beams must be >= n_best"
            )

        started = time.perf_counter()

        used_device = self._device
        fallback_reason = None

        try:
            generated = self._generate_once(
                image,
                num_beams=num_beams,
                n_best=n_best,
                max_output_length=max_output_length,
            )

        except RuntimeError as exc:

            if (
                self._device == "mps"
                and self.allow_cpu_fallback
                and self.model is not None
            ):
                fallback_reason = str(exc)

                # Move already-materialized model to CPU.
                self._device = "cpu"
                used_device = "cpu"

                self.model.to("cpu")

                generated = self._generate_once(
                    image,
                    num_beams=num_beams,
                    n_best=n_best,
                    max_output_length=max_output_length,
                )

            else:
                raise

        decoded = self.processor.batch_decode(
            generated.sequences,
            skip_special_tokens=True,
        )

        sequence_scores = None

        if (
            getattr(
                generated,
                "sequences_scores",
                None,
            )
            is not None
        ):
            sequence_scores = (
                generated.sequences_scores
                .detach()
                .float()
                .cpu()
                .tolist()
            )

        if (
            sequence_scores is not None
            and len(sequence_scores)
            == len(decoded)
        ):
            relative_scores = _softmax(
                sequence_scores
            )
        else:
            relative_scores = [
                None
                for _ in decoded
            ]

        rows: List[Dict[str, Any]] = []

        for index, text in enumerate(decoded):

            clean_text = " ".join(
                (text or "")
                .strip()
                .split()
            )

            rows.append(
                {
                    "raw_text": clean_text,
                    "relative_score": (
                        relative_scores[index]
                    ),
                    "sequence_score": (
                        None
                        if sequence_scores is None
                        else float(
                            sequence_scores[index]
                        )
                    ),
                }
            )

        # --------------------------------------------------------------
        # Merge exact duplicate generated hypotheses.
        # --------------------------------------------------------------

        merged: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []

        for row in rows:
            text = row["raw_text"]

            if text not in merged:
                order.append(text)

                merged[text] = {
                    "raw_text": text,
                    "relative_score": (
                        0.0
                        if row["relative_score"]
                        is not None
                        else None
                    ),
                    "sequence_score": (
                        row["sequence_score"]
                    ),
                }

            if (
                row["relative_score"]
                is not None
            ):
                merged[text][
                    "relative_score"
                ] += float(
                    row["relative_score"]
                )

            if (
                row["sequence_score"]
                is not None
            ):
                current = merged[text][
                    "sequence_score"
                ]

                score = float(
                    row["sequence_score"]
                )

                if (
                    current is None
                    or score > current
                ):
                    merged[text][
                        "sequence_score"
                    ] = score

        unique_rows = [
            merged[text]
            for text in order
        ]

        unique_rows.sort(
            key=lambda row: (
                -1.0
                if row["relative_score"]
                is None
                else float(
                    row["relative_score"]
                )
            ),
            reverse=True,
        )

        # --------------------------------------------------------------
        # Re-normalize probability mass after duplicate merging.
        # --------------------------------------------------------------

        known_relative_scores = [
            float(
                row["relative_score"]
            )
            for row in unique_rows
            if row["relative_score"]
            is not None
        ]

        if known_relative_scores:

            total = sum(
                known_relative_scores
            )

            if total > 0:

                for row in unique_rows:

                    if (
                        row["relative_score"]
                        is not None
                    ):
                        row[
                            "relative_score"
                        ] = (
                            float(
                                row[
                                    "relative_score"
                                ]
                            )
                            / total
                        )

        # --------------------------------------------------------------
        # Convert to common provider contract.
        # --------------------------------------------------------------

        hypotheses: List[
            ProviderHypothesis
        ] = []

        for rank, row in enumerate(
            unique_rows,
            start=1,
        ):
            hypotheses.append(
                ProviderHypothesis(
                    rank=rank,
                    raw_text=row[
                        "raw_text"
                    ],
                    relative_score=(
                        None
                        if row[
                            "relative_score"
                        ]
                        is None
                        else round(
                            float(
                                row[
                                    "relative_score"
                                ]
                            ),
                            4,
                        )
                    ),
                    sequence_score=(
                        None
                        if row[
                            "sequence_score"
                        ]
                        is None
                        else round(
                            float(
                                row[
                                    "sequence_score"
                                ]
                            ),
                            6,
                        )
                    ),
                )
            )

        runtime_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        return ProviderResult(
            hypotheses=hypotheses,
            runtime_ms=round(
                runtime_ms,
                2,
            ),
            device_used=used_device,
            metadata={
                "provider_id": (
                    self.provider_id
                ),
                "model_id": (
                    self._model_id
                ),
                "processor_id": (
                    self._processor_id
                ),
                "output_script": (
                    self.output_script
                ),
                "mps_fallback_used": (
                    fallback_reason
                    is not None
                ),
                "mps_fallback_reason": (
                    fallback_reason
                ),
            },
        )


# ---------------------------------------------------------------------------
# Provider A
# ---------------------------------------------------------------------------

class TrOCRIASTBaselineProvider(
    _TrOCRProviderBase
):
    """
    Provider A.

    Model:
        Piyush3142/trocr-sanskrit-ocr

    Processor:
        same repository

    Output:
        IAST

    This remains our baseline provider.
    """

    provider_id = (
        "trocr_iast_baseline"
    )

    output_script = "iast"

    def __init__(
        self,
        model_id: str = (
            DEFAULT_TROCR_IAST_MODEL_ID
        ),
        *,
        device: str = "auto",
        allow_cpu_fallback: bool = True,
    ) -> None:

        super().__init__(
            model_id=model_id,
            processor_id=model_id,
            device=device,
            allow_cpu_fallback=(
                allow_cpu_fallback
            ),
        )


# ---------------------------------------------------------------------------
# Provider B
# ---------------------------------------------------------------------------

class TrOCRVedicDevanagariProvider(
    _TrOCRProviderBase
):
    """
    Provider B.

    Fine-tuned model:
        yzk/trocr-large-printed-vedic

    Processor/tokenizer:
        microsoft/trocr-large-printed

    Output:
        direct Devanagari

    IMPORTANT:
    This model is being evaluated as a comparison provider. Its training
    domain is printed Vedic material, so it must not be assumed to be a
    scholar-validated manuscript HTR model.
    """

    provider_id = (
        "trocr_vedic_devanagari"
    )

    output_script = "iast"

    def __init__(
        self,
        model_id: str = (
            DEFAULT_TROCR_VEDIC_MODEL_ID
        ),
        *,
        processor_id: str = (
            DEFAULT_TROCR_VEDIC_PROCESSOR_ID
        ),
        device: str = "auto",
        allow_cpu_fallback: bool = True,
    ) -> None:

        super().__init__(
            model_id=model_id,
            processor_id=processor_id,
            device=device,
            allow_cpu_fallback=(
                allow_cpu_fallback
            ),
        )


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def create_htr_provider(
    provider_name: str,
    *,
    model_id: Optional[str] = None,
    device: str = "auto",
) -> HTRProvider:
    """
    Factory used by layer5_htr.py.

    Supported providers:

        trocr_iast_baseline
        trocr_vedic_devanagari

    Compatibility alias:

        trocr_baseline
            -> trocr_iast_baseline
    """

    provider_name = (
        provider_name
        .strip()
        .lower()
    )

    if provider_name in {
        "trocr_baseline",
        "trocr_iast_baseline",
    }:
        return (
            TrOCRIASTBaselineProvider(
                model_id=(
                    model_id
                    or DEFAULT_TROCR_IAST_MODEL_ID
                ),
                device=device,
                allow_cpu_fallback=True,
            )
        )

    if (
        provider_name
        == "trocr_vedic_devanagari"
    ):
        return (
            TrOCRVedicDevanagariProvider(
                model_id=(
                    model_id
                    or DEFAULT_TROCR_VEDIC_MODEL_ID
                ),
                processor_id=(
                    DEFAULT_TROCR_VEDIC_PROCESSOR_ID
                ),
                device=device,
                allow_cpu_fallback=True,
            )
        )

    raise ValueError(
        f"Unknown HTR provider "
        f"'{provider_name}'. "
        "Available providers: "
        "trocr_iast_baseline, "
        "trocr_vedic_devanagari"
    )
