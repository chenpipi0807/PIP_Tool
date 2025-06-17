from .PIP_longsize import PIP_longsize, PIP_ProportionalCrop
from .PIP_integer_calculator import PIP_IntegerCalculator
from .PIP_string_concatenation import PIP_StringConcatenation
from .PIP_image_concatenation import PIP_ImageConcatenation
from .PIP_Grayscale import PIP_Grayscale

NODE_CLASS_MAPPINGS = {
    "PIP_longsize": PIP_longsize,
    "PIP_ProportionalCrop": PIP_ProportionalCrop,
    "PIP_IntegerCalculator": PIP_IntegerCalculator,
    "PIP_StringConcatenation": PIP_StringConcatenation,
    "PIP_图像联结": PIP_ImageConcatenation,
    "PIP_Grayscale": PIP_Grayscale
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PIP_longsize": "PIP 长边调整",
    "PIP_ProportionalCrop": "PIP 等比例裁切",
    "PIP_IntegerCalculator": "PIP 整数计算",
    "PIP_StringConcatenation": "PIP 字符串拼接",
    "PIP_图像联结": "PIP 图像联结",
    "PIP_Grayscale": "PIP 图像去色"
}
