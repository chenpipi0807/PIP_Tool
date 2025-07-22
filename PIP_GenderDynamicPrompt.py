import re

class PIP_GenderDynamicPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prompt": ("STRING", {
                    "multiline": True,
                    "default": "一个人喝咖啡"
                }),
                "male_prefix": ("STRING", {
                    "multiline": True,
                    "default": "八块腹肌"
                }),
                "female_prefix": ("STRING", {
                    "multiline": True,
                    "default": "优雅气质"
                }),
                "gender_condition": ("STRING", {
                    "default": "男"
                }),
                "position": (["前缀", "后缀"], {
                    "default": "后缀"
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_prompt",)
    FUNCTION = "combine_prompt"
    CATEGORY = "PIP_Tool"
    
    def combine_prompt(self, base_prompt, male_prefix, female_prefix, gender_condition, position):
        """
        根据性别条件动态组合prompt
        """
        # 标准化性别判断
        gender_condition = gender_condition.lower()
        is_male = gender_condition in ["男", "male", "man"]
        is_female = gender_condition in ["女", "female", "woman"]
        
        # 选择对应的起手式
        if is_male:
            selected_prefix = male_prefix.strip()
            gender_type = "男性"
        elif is_female:
            selected_prefix = female_prefix.strip()
            gender_type = "女性"
        else:
            # 默认情况，不应该发生
            selected_prefix = ""
            gender_type = "未知"
        
        # 清理基础prompt
        base_prompt = base_prompt.strip()
        
        # 如果起手式为空，直接返回基础prompt
        if not selected_prefix:
            print(f"PIP_GenderDynamicPrompt: {gender_type}起手式为空，返回原始prompt")
            return (base_prompt,)
        
        # 根据位置参数组合prompt
        if position == "前缀":
            combined_prompt = f"{selected_prefix}, {base_prompt}"
        else:  # 后缀
            combined_prompt = f"{base_prompt}, {selected_prefix}"
        
        # 清理多余的空格和逗号
        combined_prompt = re.sub(r'\s*,\s*', ', ', combined_prompt)
        combined_prompt = re.sub(r'^,\s*|,\s*$', '', combined_prompt)
        combined_prompt = combined_prompt.strip()
        
        print(f"PIP_GenderDynamicPrompt: 检测到{gender_type}，应用起手式")
        print(f"基础prompt: {base_prompt}")
        print(f"选择的起手式: {selected_prefix}")
        print(f"位置: {position}")
        print(f"最终组合: {combined_prompt}")
        
        return (combined_prompt,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "PIP_GenderDynamicPrompt": PIP_GenderDynamicPrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PIP_GenderDynamicPrompt": "PIP 性别判断动态起手式"
}
