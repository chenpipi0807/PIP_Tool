import os
import re
from datetime import datetime
import folder_paths

class PIP_SaveTxt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "placeholder": "输入要保存的文本内容"}),
                "filename_prefix": ("STRING", {"default": "output", "placeholder": "文件名前缀"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_text_to_file"
    CATEGORY = "PIP_Tool"

    def save_text_to_file(self, text, filename_prefix):
        """
        保存文本到txt文件
        """
        try:
            # 移除中文字符和中文符号，只保留英文、数字、基本符号和空格
            # 使用正则表达式匹配ASCII字符（包括英文字母、数字、标点符号、空格等）
            english_only_text = re.sub(r'[^\x00-\x7F]', '', text)
            
            # 清理多余的空白字符
            english_only_text = re.sub(r'\s+', ' ', english_only_text).strip()
            
            # 获取ComfyUI的output目录
            output_dir = folder_paths.get_output_directory()
            
            # 生成文件名（前缀 + 时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.txt"
            
            # 完整文件路径
            file_path = os.path.join(output_dir, filename)
            
            # 保存文件，使用UTF-8编码
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(english_only_text)
            
            print(f"[PIP_SaveTxt] 文件已保存: {file_path}")
            print(f"[PIP_SaveTxt] 原始文本长度: {len(text)}")
            print(f"[PIP_SaveTxt] 处理后文本长度: {len(english_only_text)}")
            
            return (file_path,)
            
        except Exception as e:
            print(f"[PIP_SaveTxt] 保存文件时发生错误: {str(e)}")
            return (f"Error: {str(e)}",)

    @classmethod
    def IS_CHANGED(cls, text, filename_prefix):
        # 每次都重新执行，因为包含时间戳
        return float("nan")
