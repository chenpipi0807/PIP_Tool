import re
import random


class PIP_PolyGenderJudgment:
    """
    POLY-性别判断节点
    根据输入内容判断性别并输出对应的lora名字和动态提示词
    
    性别判断规则:
    - 包含完整单词"male"（不区分大小写）→ 男的
    - 包含完整单词"female"（不区分大小写）→ 女的
    - 都不包含或都包含 → 其他
    
    输出:
    - lora: 对应性别的lora名字
    - dynamic_prompt: 起手式 + 动态提示词的组合
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_content": ("STRING", {
                    "multiline": True,
                    "default": "a beautiful woman portrait"
                }),
                "male_lora": ("STRING", {
                    "default": "male_lora_v1"
                }),
                "female_lora": ("STRING", {
                    "default": "female_lora_v1"
                }),
                "other_lora": ("STRING", {
                    "default": "other_lora_v1"
                }),
                "dynamic_prompt_male": ("STRING", {
                    "multiline": True,
                    "default": "handsome man, masculine features"
                }),
                "dynamic_prompt_female": ("STRING", {
                    "multiline": True,
                    "default": "beautiful woman, elegant features"
                }),
                "dynamic_prompt_other": ("STRING", {
                    "multiline": True,
                    "default": "person, neutral features"
                }),
                "prefix_prompt_male": ("STRING", {
                    "multiline": True,
                    "default": "strong and confident"
                }),
                "prefix_prompt_female": ("STRING", {
                    "multiline": True,
                    "default": "elegant and graceful"
                }),
                "prefix_prompt_other": ("STRING", {
                    "multiline": True,
                    "default": "unique and artistic"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("lora", "prefix_prompt", "dynamic_prompt")
    FUNCTION = "judge_gender"
    CATEGORY = "PIP_Tool"
    
    def judge_gender(self, input_content, male_lora, female_lora, other_lora, 
                    dynamic_prompt_male, dynamic_prompt_female, dynamic_prompt_other,
                    prefix_prompt_male, prefix_prompt_female, prefix_prompt_other):
        """
        判断性别并返回对应的lora、起手式和动态提示词
        """
        # 判断性别
        gender = self._detect_gender(input_content)
        
        # 根据性别返回对应的输出
        if gender == "male":
            lora_output = male_lora
            prefix_output = prefix_prompt_male
            dynamic_output = dynamic_prompt_male
        elif gender == "female":
            lora_output = female_lora
            prefix_output = prefix_prompt_female
            dynamic_output = dynamic_prompt_female
        else:  # 其他情况
            lora_output = other_lora
            prefix_output = prefix_prompt_other
            dynamic_output = dynamic_prompt_other
        
        print(f"[PIP_PolyGenderJudgment] 输入内容: {input_content}")
        print(f"[PIP_PolyGenderJudgment] 检测到性别: {gender}")
        print(f"[PIP_PolyGenderJudgment] 输出lora: {lora_output}")
        print(f"[PIP_PolyGenderJudgment] 输出起手式: {prefix_output}")
        print(f"[PIP_PolyGenderJudgment] 输出动态提示词: {dynamic_output}")
        
        return (lora_output, prefix_output, dynamic_output)
    
    def _detect_gender(self, text):
        """
        检测文本中的性别
        
        规则:
        - 包含完整单词"male"（不区分大小写）→ male
        - 包含完整单词"female"（不区分大小写）→ female
        - 都不包含或都包含 → 其他
        
        注意: "female"包含"male"但它是一个完整的单词，所以不算包含"male"
        """
        if not text:
            return "其他"
        
        text_lower = text.lower()
        
        # 使用单词边界\b来匹配完整单词
        # \b确保匹配的是完整的单词，而不是单词的一部分
        has_male = bool(re.search(r'\bmale\b', text_lower))
        has_female = bool(re.search(r'\bfemale\b', text_lower))
        
        print(f"[PIP_PolyGenderJudgment] 文本分析: '{text}'")
        print(f"[PIP_PolyGenderJudgment] 包含'male': {has_male}")
        print(f"[PIP_PolyGenderJudgment] 包含'female': {has_female}")
        
        # 判断逻辑
        if has_male and has_female:
            # 都包含 → 其他
            return "其他"
        elif has_male:
            # 只包含male → male
            return "male"
        elif has_female:
            # 只包含female → female
            return "female"
        else:
            # 都不包含 → 其他
            return "其他"
    

