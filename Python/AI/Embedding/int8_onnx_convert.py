from pathlib import Path
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import QuantizationConfig
from onnxruntime.quantization import QuantType, QuantFormat, QuantizationMode

MODEL_DIR = Path("onnx_models/multilingual-e5-base")

quantizer = ORTQuantizer.from_pretrained(MODEL_DIR, file_name="model.onnx")

qconfig = QuantizationConfig(
    is_static=False,
    format=QuantFormat.QOperator,
    mode=QuantizationMode.IntegerOps,
    activations_dtype=QuantType.QUInt8,
    weights_dtype=QuantType.QInt8,
    per_channel=False,
    reduce_range=False,
)

quantizer.quantize(save_dir=MODEL_DIR, quantization_config=qconfig)
print("완료!")


# multilingual-e5-base 모델
# https://huggingface.co/intfloat/multilingual-e5-base/tree/main/onnx

# 1. config.json
# 2. tokenizor.json
# 3. tokenizer_config.json
# 4. special_tokens_map.json
# 5. sentencepiece.bpe.model
# 6. model.onnx -> int8양자화 필요 int8_onnx_convert.py (vm), int8_convert.py (mac)
