from .PIP_longsize import PIP_longsize, PIP_ProportionalCrop
from .PIP_integer_calculator import PIP_IntegerCalculator
from .PIP_string_concatenation import PIP_StringConcatenation
from .PIP_image_concatenation import PIP_ImageConcatenation
from .PIP_seamless_concatenation import PIP_SeamlessConcatenation
from .PIP_Grayscale import PIP_Grayscale
from .PIP_HeadCrop import PIP_HeadCrop
from .PIP_PuzzleTool import PIP_PuzzleTool, PIP_DynamicPuzzleSelection
from .PIP_PuzzleTool_Enhanced import PIP_PuzzleTool_Enhanced, PIP_DynamicPuzzleSelection_Enhanced
from .PIP_SearchReplace import PIP_SearchReplace
from .PIP_Kontext import PIP_Kontext
from .PIP_workflow_extractor import PIP_WorkflowExtractor
from .PIP_load_image_url import PIP_LoadImageURL
from .PIP_load_json_url import PIP_LoadJSONURL
from .PIP_batch_json_extractor import PIP_batchJSONExtractor
from .PIP_DualRoleJudgmentSystem import PIP_DualRoleJudgmentSystem
from .PIP_novel_batch_validator import PIP_NovelBatchValidator
from .PIP_TextToImage import PIP_TextToImage
from .PIP_RGBA_to_RGB import PIP_RGBA_to_RGB, PIP_RGBAtoRGB
from .PIP_SaveTxt import PIP_SaveTxt
from .PIP_Pixelate import PIP_Pixelate, PIP_PixelateAdvanced
from .PIP_DynamicPrompt import PIP_DynamicPrompt
from .PIP_EdgeExpand import PIP_EdgeExpand
from .PIP_BrightnessAnalysis import PIP_BrightnessAnalysis
from .PIP_BrightnessCorrection import PIP_BrightnessCorrection

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
    "PIP_PuzzleTool_Enhanced": PIP_PuzzleTool_Enhanced,
    "PIP_DynamicPuzzleSelection_Enhanced": PIP_DynamicPuzzleSelection_Enhanced,
    "PIP_SearchReplace": PIP_SearchReplace,
    "PIP_Kontext": PIP_Kontext,
    "PIP_WorkflowExtractor": PIP_WorkflowExtractor,
    "PIP_LoadImageURL": PIP_LoadImageURL,
    "PIP_LoadJSONURL": PIP_LoadJSONURL,
    "PIP_batchJSONExtractor": PIP_batchJSONExtractor,
    "PIP_DualRoleJudgmentSystem": PIP_DualRoleJudgmentSystem,
    "PIP_NovelBatchValidator": PIP_NovelBatchValidator,
    "PIP_TextToImage": PIP_TextToImage,
    "PIP_RGBA_to_RGB": PIP_RGBA_to_RGB,
    "PIP_RGBAtoRGB": PIP_RGBAtoRGB,
    "PIP_SaveTxt": PIP_SaveTxt,
    "PIP_Pixelate": PIP_Pixelate,
    "PIP_PixelateAdvanced": PIP_PixelateAdvanced,
    "PIP_DynamicPrompt": PIP_DynamicPrompt,
    "PIP_EdgeExpand": PIP_EdgeExpand,
    "PIP_BrightnessAnalysis": PIP_BrightnessAnalysis,
    "PIP_BrightnessCorrection": PIP_BrightnessCorrection
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
    "PIP_PuzzleTool_Enhanced": "PIP 拼图工具 (增强版支持URL)",
    "PIP_DynamicPuzzleSelection_Enhanced": "PIP 拼图动态选择 (增强版支持URL)",
    "PIP_SearchReplace": "PIP 搜索替换",
    "PIP_Kontext": "PIP Kontext AI图像编辑",
    "PIP_WorkflowExtractor": "PIP 工作流变量提取",
    "PIP_LoadImageURL": "PIP URL图像加载",
    "PIP_LoadJSONURL": "PIP URL-JSON加载",
    "PIP_batchJSONExtractor": "PIP 批次JSON提取器",
    "PIP_DualRoleJudgmentSystem": "PIP 双角色判断系统",
    "PIP_NovelBatchValidator": "novel小说批次验证专用",
    "PIP_TextToImage": "PIP 文本转图像",
    "PIP_RGBA_to_RGB": "PIP RGBA转RGB",
    "PIP_RGBAtoRGB": "PIP RGBA转RGB (简化版)",
    "PIP_SaveTxt": "PIP 保存文本",
    "PIP_Pixelate": "PIP 像素化",
    "PIP_PixelateAdvanced": "PIP 像素化 (高级版)",
    "PIP_DynamicPrompt": "PIP 动态提示词",
    "PIP_EdgeExpand": "PIP 边缘扩图",
    "PIP_BrightnessAnalysis": "PIP 亮度检测",
    "PIP_BrightnessCorrection": "PIP 亮度补偿"
}
