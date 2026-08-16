import torch
import transformers

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

MODEL = "yzk/trocr-large-printed-vedic"
BASE_PROCESSOR = "microsoft/trocr-large-printed"

print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("MPS available:", torch.backends.mps.is_available())

print("\n1. Loading full TrOCR processor from Microsoft base...")
processor = TrOCRProcessor.from_pretrained(BASE_PROCESSOR)

print("Processor OK:", type(processor).__name__)
print("Tokenizer OK:", type(processor.tokenizer).__name__)
print("Tokenizer vocab size:", len(processor.tokenizer))
print("Image processor:", type(processor.image_processor).__name__)

print("\n2. Loading Vedic fine-tuned model...")
model = VisionEncoderDecoderModel.from_pretrained(
    MODEL,
    low_cpu_mem_usage=False,
)

print("Model class:", type(model).__name__)
print("Decoder vocab size:", model.config.decoder.vocab_size)
print("First parameter device:", next(model.parameters()).device)

meta_params = [
    name
    for name, param in model.named_parameters()
    if param.device.type == "meta"
]

meta_buffers = [
    name
    for name, buffer in model.named_buffers()
    if buffer.device.type == "meta"
]

print("\nMeta parameters:", len(meta_params))
if meta_params:
    print("First meta parameters:")
    for name in meta_params[:20]:
        print("  PARAM:", name)

print("Meta buffers:", len(meta_buffers))
if meta_buffers:
    print("First meta buffers:")
    for name in meta_buffers[:20]:
        print("  BUFFER:", name)

if len(processor.tokenizer) != model.config.decoder.vocab_size:
    print(
        "\nSTOP: TOKENIZER/MODEL VOCAB SIZE MISMATCH:",
        len(processor.tokenizer),
        "vs",
        model.config.decoder.vocab_size,
    )
elif meta_params or meta_buffers:
    print("\nSTOP: MODEL STILL CONTAINS META TENSORS")
else:
    print("\n3. CPU materialization successful.")
    print("Tokenizer/model vocabulary sizes match.")

    if torch.backends.mps.is_available():
        print("\n4. Moving model to MPS...")
        model = model.to("mps")
        print("Model device:", next(model.parameters()).device)
        print("\nPROVIDER B MODEL + BASE PROCESSOR LOAD TEST PASSED")
    else:
        print("\nMPS unavailable; CPU load test passed.")
