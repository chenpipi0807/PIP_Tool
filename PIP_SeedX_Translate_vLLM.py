import re
import os
from typing import Optional, List, Tuple

# 尝试导入vLLM（官方推荐）
try:
    from vllm import LLM, SamplingParams
    # 注意：BeamSearchParams在vLLM中不存在，使用SamplingParams的beam search功能
    VLLM_AVAILABLE = True
    print("vLLM is available for this node")
except ImportError as e:
    print(f"vLLM not available: {e}")
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

# 导入transformers作为备用
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Transformers not available")
    TRANSFORMERS_AVAILABLE = False

class PIP_SeedX_Translate_vLLM:
    """
    Seed-X 翻译节点 - 基于vLLM的官方实现
    完全按照官方README实现，支持自动语言检测和智能混合语言处理
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.use_vllm = VLLM_AVAILABLE
        self.current_model_path = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, how are you today?"
                }),
                "target_language": (["中文", "English", "日本語", "한국어", "Français", "Deutsch", "Español", "Italiano", "Русский"], {
                    "default": "中文"
                }),
                "max_length": ("INT", {
                    "default": 512,
                    "min": 50,
                    "max": 2048
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "do_sample": ("BOOLEAN", {
                    "default": False
                }),
                "num_beams": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8
                }),
                "enable_cot": ("BOOLEAN", {
                    "default": False
                })
            },
            "optional": {
                "model_path": ("STRING", {
                    "default": "models/Seed-X-Instruct-7B"
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "translate_text"
    CATEGORY = "PIP_Tool"
    
    def _get_language_code(self, language: str) -> str:
        """获取语言代码"""
        language_map = {
            "中文": "zh",
            "English": "en", 
            "日本語": "ja",
            "한국어": "ko",
            "Français": "fr",
            "Deutsch": "de",
            "Español": "es",
            "Italiano": "it",
            "Русский": "ru"
        }
        return language_map.get(language, "zh")
    
    def _detect_language(self, text: str) -> str:
        """简单的语言检测"""
        # 检测中文字符
        if re.search(r'[\u4e00-\u9fff]', text):
            return "zh"
        # 检测日文字符
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return "ja"
        # 检测韩文字符
        elif re.search(r'[\uac00-\ud7af]', text):
            return "ko"
        # 检测俄文字符
        elif re.search(r'[\u0400-\u04ff]', text):
            return "ru"
        # 默认为英文
        else:
            return "en"
    
    def _get_language_name(self, code: str) -> str:
        """根据代码获取语言名称"""
        code_to_name = {
            "zh": "Chinese",
            "en": "English",
            "ja": "Japanese", 
            "ko": "Korean",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "it": "Italian",
            "ru": "Russian"
        }
        return code_to_name.get(code, "English")
    
    def _split_mixed_language_text(self, text: str) -> List[Tuple[str, str]]:
        """
        智能分割混合语言文本
        返回: [(文本片段, 语言代码), ...]
        """
        segments = []
        current_segment = ""
        current_lang = None
        
        for char in text:
            char_lang = self._detect_language(char)
            
            if current_lang is None:
                current_lang = char_lang
                current_segment = char
            elif char_lang == current_lang or char_lang == "en":
                current_segment += char
            else:
                if current_segment.strip():
                    segments.append((current_segment.strip(), current_lang))
                current_segment = char
                current_lang = char_lang
        
        if current_segment.strip():
            segments.append((current_segment.strip(), current_lang))
        
        return segments
    
    def _resolve_model_path(self, model_path: str) -> str:
        """解析模型路径 - 优先使用脚本相对路径"""
        # 如果是绝对路径，直接返回
        if os.path.isabs(model_path):
            return model_path
            
        # 直接使用脚本相对路径
        script_relative_path = os.path.join(os.path.dirname(__file__), "models", "Seed-X-Instruct-7B")
        if os.path.exists(script_relative_path):
            print(f"Using script relative path: {script_relative_path}")
            return script_relative_path
            
        # 备用方案：尝试其他路径
        possible_paths = [
            os.path.join(os.path.dirname(__file__), model_path),  # 相对于当前文件
            os.path.join(os.getcwd(), model_path),  # 当前工作目录
            model_path,  # 原始路径
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                print(f"Found model at: {abs_path}")
                return abs_path
                
        print(f"Warning: Model not found in any expected location")
        return model_path  # 如果都找不到，返回原始路径
    
    def _load_model(self, model_path: str) -> bool:
        """加载模型 - 优先使用vLLM，transformers作为备用"""
        if self.current_model_path == model_path and (self.model or self.tokenizer):
            return True
            
        try:
            resolved_path = self._resolve_model_path(model_path)
            print(f"Loading Seed-X model from: {resolved_path}")
            
            if VLLM_AVAILABLE:
                try:
                    # 优先使用vLLM
                    print("Loading model with vLLM (official method)...")
                    self.model = LLM(
                        model=resolved_path,
                        max_num_seqs=512,
                        enable_prefix_caching=True,
                        gpu_memory_utilization=0.9
                    )
                    self.use_vllm = True
                    self.current_model_path = model_path
                    print("Seed-X model loaded successfully with vLLM!")
                    return True
                except Exception as e:
                    print(f"vLLM loading failed: {str(e)}")
                    self.model = None
            
            if TRANSFORMERS_AVAILABLE:
                try:
                    # 检查必需的tokenizer文件
                    import os
                    tokenizer_json_path = os.path.join(resolved_path, "tokenizer.json")
                    if os.path.exists(tokenizer_json_path):
                        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if not content:
                                print(f"Error: tokenizer.json is empty at {tokenizer_json_path}")
                                print("Please download the complete model from Hugging Face")
                                raise ValueError("Empty tokenizer.json file")
                    
                    # 使用transformers作为备用
                    print("Loading model with transformers (fallback)...")
                    self.tokenizer = AutoTokenizer.from_pretrained(resolved_path)
                    
                    # 设置padding token
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        print("Set pad_token to eos_token")
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        resolved_path,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                    self.use_vllm = False
                    self.current_model_path = model_path
                    print("Seed-X model loaded successfully with transformers!")
                    return True
                except Exception as e:
                    print(f"Transformers loading failed: {str(e)}")
                    if "tokenizer.json" in str(e) or "Expecting value" in str(e):
                        print("\n🚨 Model files appear to be incomplete or corrupted!")
                        print("Please download the complete Seed-X model from:")
                        print("https://huggingface.co/Seed-X/Seed-X-Instruct-7B")
                        print("\nRequired files:")
                        print("- config.json")
                        print("- tokenizer.json (must not be empty)")
                        print("- tokenizer_config.json")
                        print("- special_tokens_map.json")
                        print("- model.safetensors")
                        print("- generation_config.json")
                    self.model = None
                    self.tokenizer = None
            
            print("Error: No compatible library available for model loading")
            return False
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
            self.tokenizer = None
            return False
    
    def _create_translation_prompt(self, text: str, source_lang: str, target_lang: str, enable_cot: bool = False) -> str:
        """创建翻译提示词 - 按照官方格式"""
        source_name = self._get_language_name(source_lang)
        target_name = self._get_language_name(target_lang)
        
        if enable_cot:
            # CoT格式
            prompt = f"Translate the following {source_name} text into {target_name} and explain it in detail:\n{text} <{target_lang}>"
        else:
            # 标准格式
            prompt = f"Translate the following {source_name} text into {target_name}:\n{text} <{target_lang}>"
        
        return prompt
    
    def _generate_translation(self, prompt: str, max_length: int, temperature: float, do_sample: bool, num_beams: int = 4) -> str:
        """生成翻译结果 - 优先使用vLLM，transformers作为备用"""
        try:
            if self.use_vllm and VLLM_AVAILABLE and hasattr(self, 'model') and self.model:
                # 使用vLLM生成
                if do_sample:
                    # Sampling模式
                    decoding_params = SamplingParams(
                        temperature=temperature,
                        max_tokens=max_length,
                        skip_special_tokens=True
                    )
                else:
                    # Beam Search模式（使用SamplingParams的beam search功能）
                    decoding_params = SamplingParams(
                        temperature=0.0,
                        max_tokens=max_length,
                        best_of=num_beams,
                        use_beam_search=True,
                        skip_special_tokens=True
                    )
                
                # 生成翻译
                results = self.model.generate([prompt], decoding_params)
                responses = [res.outputs[0].text.strip() for res in results]
                
                return responses[0] if responses and responses[0] else "No translation generated"
            
            elif not self.use_vllm and TRANSFORMERS_AVAILABLE and hasattr(self, 'tokenizer') and self.tokenizer:
                # 使用transformers作为备用
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    if do_sample:
                        # Sampling模式
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_length,
                            temperature=temperature,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    else:
                        # Beam Search模式
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_length,
                            num_beams=num_beams,
                            temperature=temperature,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                
                # 解码结果
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][input_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                return generated_text.strip() if generated_text else "No translation generated"
            
            else:
                return "Error: No compatible model loaded"
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return f"Translation error: {str(e)}"
    
    def translate_text(self, text: str, target_language: str, max_length: int = 512, 
                      temperature: float = 0.7, do_sample: bool = False, num_beams: int = 4,
                      enable_cot: bool = False, model_path: str = "models/Seed-X-Instruct-7B") -> tuple:
        """主要翻译函数"""
        print("got prompt")
        
        # 加载模型
        if self.model is None and self.tokenizer is None:
            if not self._load_model(model_path):
                return ("Failed to load model",)
        
        target_code = self._get_language_code(target_language)
        
        # 检测混合语言并分段处理
        segments = self._split_mixed_language_text(text)
        translated_segments = []
        
        for segment_text, detected_lang in segments:
            if detected_lang == target_code:
                # 如果已经是目标语言，直接添加
                translated_segments.append(segment_text)
            else:
                # 创建提示词
                prompt = self._create_translation_prompt(segment_text, detected_lang, target_code, enable_cot)
                
                # 生成翻译
                translation = self._generate_translation(prompt, max_length, temperature, do_sample, num_beams)
                translated_segments.append(translation)
        
        # 合并结果
        final_translation = " ".join(translated_segments)
        return (final_translation,)

# ComfyUI节点映射
NODE_CLASS_MAPPINGS = {
    "PIP_SeedX_Translate_vLLM": PIP_SeedX_Translate_vLLM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PIP_SeedX_Translate_vLLM": "Seed-X Translate (vLLM)"
}
