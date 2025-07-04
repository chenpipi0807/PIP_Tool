import re
import random


class PIP_DynamicPrompt:
    """
    动态提示词节点 - 随机选择{}内的选项
    输入格式: "1 man with{red|green|blue} hair and {tall|short} body"
    输出示例: "1 man with green hair and tall body"
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "1 man with{red|green|blue} hair and {red/green/blue} eyes, wearing {casual {t-shirt|hoodie}|formal {suit|blazer}}"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999999
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dynamic_prompt",)
    FUNCTION = "generate_dynamic_prompt"
    CATEGORY = "PIP_Tool"
    
    def generate_dynamic_prompt(self, prompt, seed):
        """
        处理动态提示词，随机选择{}内的选项
        支持嵌套花括号和多种分隔符 (|, /, \)
        """
        # 设置随机种子确保可重复性
        random.seed(seed)
        
        original_prompt = prompt
        
        # 递归处理嵌套的花括号，从最内层开始
        def process_nested_braces(text):
            # 找到最内层的花括号（不包含其他花括号的）
            pattern = r'\{([^{}]+)\}'
            
            def replace_innermost(match):
                options_str = match.group(1)
                # 支持多种分隔符: |, /, \
                # 使用正则表达式分割，支持这三种分隔符
                options = re.split(r'[|/\\]', options_str)
                options = [opt.strip() for opt in options if opt.strip()]
                
                if options:
                    return random.choice(options)
                else:
                    return options_str  # 如果没有有效选项，返回原文
            
            # 持续处理直到没有花括号为止
            while re.search(pattern, text):
                text = re.sub(pattern, replace_innermost, text)
            
            return text
        
        # 处理所有嵌套的花括号
        result = process_nested_braces(prompt)
        
        print(f"[PIP_DynamicPrompt] 原始提示词: {original_prompt}")
        print(f"[PIP_DynamicPrompt] 随机结果: {result}")
        print(f"[PIP_DynamicPrompt] 使用种子: {seed}")
        
        return (result,)
