import re
import random


class PIP_DynamicPrompt:
    """
    动态提示词节点 - 随机选择{}内的选项
    输入格式: "1 man with{red|green|blue} hair and {tall|short} body"
    输出示例: "1 man with green hair and tall body"
    
    高优先级输入功能:
    - 如果高优先级输入非空且有效，直接输出高优先级内容
    - 如果高优先级输入为空或无效，执行动态提示词处理
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "1 man with{red|green|blue} hair and {red/green/blue} eyes, wearing {casual {t-shirt|hoodie}|formal {suit|blazer}}"
                }),
                "high_priority_input": ("STRING", {
                    "multiline": True,
                    "default": ""
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
    
    def generate_dynamic_prompt(self, prompt, high_priority_input, seed):
        """
        处理动态提示词，随机选择{}内的选项
        支持嵌套花括号和多种分隔符 (|, /, \)
        
        高优先级输入逻辑:
        - 如果高优先级输入有效（非空且符合条件），直接返回
        - 否则执行动态提示词处理
        """
        # 首先检查高优先级输入是否有效
        if self._is_high_priority_valid(high_priority_input):
            print(f"[PIP_DynamicPrompt] 使用高优先级输入: {high_priority_input}")
            return (high_priority_input,)
        
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
    
    def _is_high_priority_valid(self, high_priority_input):
        """
        判断高优先级输入是否有效
        
        视为空的情况:
        1. 空字符串
        2. 仅包含阿拉伯数字或符号
        3. 空格与标点的组合
        4. 只有一个单词+标点符号+空格
        
        视为非空的情况:
        - 超过两个单词
        """
        if not high_priority_input or not high_priority_input.strip():
            return False
        
        text = high_priority_input.strip()
        
        # 1. 仅包含阿拉伯数字或符号（非字母字符）
        if re.match(r'^[^a-zA-Z\u4e00-\u9fff]+$', text):
            print(f"[PIP_DynamicPrompt] 高优先级输入无效: 仅包含数字或符号 '{text}'")
            return False
        
        # 2. 空格与标点的组合（移除所有字母和中文后，如果只剩下空格和标点）
        text_without_letters = re.sub(r'[a-zA-Z\u4e00-\u9fff]', '', text)
        if text_without_letters.strip() == text.strip():
            print(f"[PIP_DynamicPrompt] 高优先级输入无效: 空格与标点组合 '{text}'")
            return False
        
        # 3. 提取有效单词（字母组成的词汇，包括中文）
        words = re.findall(r'[a-zA-Z\u4e00-\u9fff]+', text)
        
        # 只有一个单词视为无效
        if len(words) <= 1:
            print(f"[PIP_DynamicPrompt] 高优先级输入无效: 只有{len(words)}个单词 '{text}'")
            return False
        
        # 超过两个单词视为有效
        print(f"[PIP_DynamicPrompt] 高优先级输入有效: {len(words)}个单词 '{text}'")
        return True
