import torch
import transformers

from transformers import (
    ViTImageProcessor,
    AutoTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

MODEL = "yzk/trocr-large-printed-vedic"
BASE_PROCESSOR = "microsoft/trocr-large-printed"

print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("MPS available:", torch.backends.mps.is_available())

print("\n1. Loading image processor from Microsoft TrOCR base...")
image_processor = ViTImageProcessor.from_pretrained(BASE_PROCESSOR)
print("Image processor OK:", type(image_processor).__name__)

print("\n2. Loading Vedic model tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    use_fast=True,
)
print("Tokenizer OK:", type(tokenizer).__name__)

print("\n3. Constructing TrOCR processor manually...")
processor = TrOCRProcessor(
    image_processor=image_processor,
    tokenizer=tokenizer,
)
print("Processor OK:", type(processor).__name__)

print("\n4. Loading Provider-B model...")
model = VisionEncoderDecoderModel.from_pretrained(
    MODEL,
    low_cpu_mem_usage=False,
)

print("Model class:", type(model).__name__)
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

if meta_params or meta_buffers:
    print("\nSTOP: MODEL STILL CONTAINS META TENSORS")
else:
    print("\n5. CPU materialization successful.")

    if torch.backends.mps.is_available():
        print("Moving model to MPS...")
        model = model.to("mps")
        print("Model device:", next(model.parameters()).device)
        print("\nPROVIDER B MODEL + PROCESSOR LOAD TEST PASSED")
    else:
        print("MPS unavailable; CPU load test passed.")
