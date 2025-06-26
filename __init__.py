from .PIP_longsize import PIP_longsize, PIP_ProportionalCrop
from .PIP_integer_calculator import PIP_IntegerCalculator
from .PIP_string_concatenation import PIP_StringConcatenation
from .PIP_image_concatenation import PIP_ImageConcatenation
from .PIP_seamless_concatenation import PIP_SeamlessConcatenation
from .PIP_Grayscale import PIP_Grayscale
from .PIP_HeadCrop import PIP_HeadCrop
from .PIP_PuzzleTool import PIP_PuzzleTool, PIP_DynamicPuzzleSelection
from .PIP_SearchReplace import PIP_SearchReplace
from .PIP_Kontext import PIP_Kontext
from .PIP_workflow_extractor import PIP_WorkflowExtractor
from .PIP_load_image_url import PIP_LoadImageURL
from .PIP_load_json_url import PIP_LoadJSONURL
from .PIP_batch_json_extractor import PIP_batchJSONExtractor
from .PIP_DualRoleJudgmentSystem import PIP_DualRoleJudgmentSystem

NODE_CLASS_MAPPINGS = {
    "PIP_longsize": PIP_longsize,
    "PIP_ProportionalCrop": PIP_ProportionalCrop,
    "PIP_IntegerCalculator": PIP_IntegerCalculator,
    "PIP_StringConcatenation": PIP_StringConcatenation,
    "PIP_图像联结": PIP_ImageConcatenation,
    "PIP_无缝拼接": PIP_SeamlessConcatenation,
    "PIP_Grayscale": PIP_Grayscale,
    "PIP_HeadCrop": PIP_HeadCrop,
    "PIP_PuzzleTool": PIP_PuzzleTool,
    "PIP_DynamicPuzzleSelection": PIP_DynamicPuzzleSelection,
    "PIP_SearchReplace": PIP_SearchReplace,
    "PIP_Kontext": PIP_Kontext,
    "PIP_WorkflowExtractor": PIP_WorkflowExtractor,
    "PIP_LoadImageURL": PIP_LoadImageURL,
    "PIP_LoadJSONURL": PIP_LoadJSONURL,
    "PIP_batchJSONExtractor": PIP_batchJSONExtractor,
    "PIP_DualRoleJudgmentSystem": PIP_DualRoleJudgmentSystem
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PIP_longsize": "PIP 长边调整",
    "PIP_ProportionalCrop": "PIP 等比例裁切",
    "PIP_IntegerCalculator": "PIP 整数计算",
    "PIP_StringConcatenation": "PIP 字符串拼接",
    "PIP_图像联结": "PIP 图像联结",
    "PIP_无缝拼接": "PIP 无缝拼接",
    "PIP_Grayscale": "PIP 图像去色",
    "PIP_HeadCrop": "PIP 人脸检测",
    "PIP_PuzzleTool": "PIP 拼图工具",
    "PIP_DynamicPuzzleSelection": "PIP 拼图动态选择",
    "PIP_SearchReplace": "PIP 搜索替换",
    "PIP_Kontext": "PIP Kontext AI图像编辑",
    "PIP_WorkflowExtractor": "PIP 工作流变量提取",
    "PIP_LoadImageURL": "PIP URL图像加载",
    "PIP_LoadJSONURL": "PIP URL-JSON加载",
    "PIP_batchJSONExtractor": "PIP 批次JSON提取器",
    "PIP_DualRoleJudgmentSystem": "PIP 双角色判断系统"
}
