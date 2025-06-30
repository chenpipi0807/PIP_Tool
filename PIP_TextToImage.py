import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from pathlib import Path
import textwrap

class PIP_TextToImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "在这里输入小说简介或其他文本内容...", "multiline": True}),
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64
                }),
                "height": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 4096,
                    "step": 64
                }),
                "font_size": ("INT", {
                    "default": 32,
                    "min": 12,
                    "max": 120,
                    "step": 2
                }),
                "margin_percent": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.05,
                    "max": 0.3,
                    "step": 0.01
                }),
                "line_spacing": ("FLOAT", {
                    "default": 1.2,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "text_to_image"
    CATEGORY = "图像处理"

    def text_to_image(self, text, width, height, font_size, margin_percent, line_spacing):
        # Find and load the font
        font_path = self._find_font_path("Arial_Unicode.ttf")
        font = None
        
        if font_path:
            try:
                font = ImageFont.truetype(font_path, font_size)
                print(f"Successfully loaded font: {font_path}")
            except Exception as e:
                print(f"Error loading font {font_path}: {e}")
        
        # If font loading failed, try to load default font with size
        if font is None:
            try:
                # Try to load a default font with size
                font = ImageFont.load_default()
                print("Using default font")
            except Exception as e:
                print(f"Error loading default font: {e}")
                # Last resort: create a basic font
                font = ImageFont.load_default()

        # Create black background image
        image = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Calculate margins
        margin_x = int(width * margin_percent)
        margin_y = int(height * margin_percent)
        
        # Available text area
        text_width = width - 2 * margin_x
        text_height = height - 2 * margin_y

        # Clean and prepare text
        text = text.strip()
        if not text:
            text = "请输入文本内容"

        # Calculate optimal text layout
        wrapped_lines = self._wrap_text_to_fit(text, font, text_width, text_height, line_spacing)
        
        if not wrapped_lines:
            # If text doesn't fit, try smaller font
            smaller_font_size = max(12, int(font_size * 0.8))
            try:
                smaller_font = ImageFont.truetype(font_path, smaller_font_size)
                wrapped_lines = self._wrap_text_to_fit(text, smaller_font, text_width, text_height, line_spacing)
                font = smaller_font
                print(f"Font size reduced to {smaller_font_size} to fit text")
            except:
                # Use original font and truncate if necessary
                wrapped_lines = self._wrap_text_simple(text, font, text_width)

        # Calculate total text block height
        line_height = int(font_size * line_spacing)
        total_text_height = len(wrapped_lines) * line_height

        # Center the text block vertically
        start_y = margin_y + (text_height - total_text_height) // 2
        
        # Draw each line
        current_y = start_y
        for line in wrapped_lines:
            if line.strip():  # Skip empty lines
                # Get text width for centering
                try:
                    text_bbox = draw.textbbox((0, 0), line, font=font)
                    line_width = text_bbox[2] - text_bbox[0]
                except AttributeError:
                    # Fallback for older PIL versions
                    line_width = font.getlength(line)
                
                # Center horizontally
                x = margin_x + (text_width - line_width) // 2
                
                # Draw the text
                draw.text((x, current_y), line, font=font, fill=(255, 255, 255))
            
            current_y += line_height

        # Convert PIL image to tensor
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # Add batch dimension
        
        return (image_tensor,)

    def _find_font_path(self, font_filename):
        """Search for the font file in various possible locations"""
        possible_paths = [
            # Relative paths
            Path("fonts") / font_filename,
            Path(__file__).parent / "fonts" / font_filename,
            Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'fonts'))) / font_filename,
            Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fonts'))) / font_filename,
            # Windows system fonts
            Path("C:/Windows/Fonts") / font_filename,
            Path("C:/Windows/Fonts/arial.ttf"),  # Fallback to Arial
            Path("C:/Windows/Fonts/simsun.ttc"),  # Chinese font
            Path("C:/Windows/Fonts/simhei.ttf"),  # Chinese font
            Path("C:/Windows/Fonts/msyh.ttc"),   # Microsoft YaHei
        ]
        
        for font_path in possible_paths:
            if font_path.exists():
                print(f"Using font: {font_path}")
                return str(font_path)
        
        print(f"Warning: No suitable font found. Using default font.")
        return None

    def _wrap_text_to_fit(self, text, font, max_width, max_height, line_spacing):
        """Wrap text to fit within given dimensions with better Chinese support"""
        # Calculate line height
        line_height = int(font.size * line_spacing)
        max_lines = max_height // line_height
        
        if max_lines <= 0:
            return []

        # Split text into paragraphs
        paragraphs = text.split('\n')
        wrapped_lines = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                wrapped_lines.append('')
                continue
            
            # For Chinese text, use character-based wrapping
            if self._contains_chinese(paragraph):
                wrapped_lines.extend(self._wrap_chinese_text(paragraph, font, max_width, max_lines - len(wrapped_lines)))
            else:
                # For English text, use word-based wrapping
                wrapped_lines.extend(self._wrap_english_text(paragraph, font, max_width, max_lines - len(wrapped_lines)))
            
            # Check if we've exceeded max lines
            if len(wrapped_lines) >= max_lines:
                return wrapped_lines[:max_lines]
        
        return wrapped_lines
    
    def _contains_chinese(self, text):
        """Check if text contains Chinese characters"""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False
    
    def _wrap_chinese_text(self, text, font, max_width, max_lines):
        """Wrap Chinese text character by character"""
        lines = []
        current_line = ""
        
        for char in text:
            test_line = current_line + char
            
            # Get actual width
            try:
                text_bbox = font.getbbox(test_line)
                actual_width = text_bbox[2] - text_bbox[0]
            except AttributeError:
                try:
                    actual_width = font.getlength(test_line)
                except AttributeError:
                    # Fallback estimation
                    actual_width = len(test_line) * font.size * 0.8
            
            if actual_width <= max_width:
                current_line = test_line
            else:
                # Current line is full, start a new line
                if current_line:
                    lines.append(current_line)
                    if len(lines) >= max_lines:
                        break
                current_line = char
        
        # Add the last line if it has content
        if current_line and len(lines) < max_lines:
            lines.append(current_line)
        
        return lines
    
    def _wrap_english_text(self, text, font, max_width, max_lines):
        """Wrap English text word by word"""
        lines = []
        words = text.split()
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            
            # Get actual width
            try:
                text_bbox = font.getbbox(test_line)
                actual_width = text_bbox[2] - text_bbox[0]
            except AttributeError:
                try:
                    actual_width = font.getlength(test_line)
                except AttributeError:
                    # Fallback estimation
                    actual_width = len(test_line) * font.size * 0.6
            
            if actual_width <= max_width:
                current_line = test_line
            else:
                # Current line is full, start a new line
                if current_line:
                    lines.append(current_line)
                    if len(lines) >= max_lines:
                        break
                current_line = word
        
        # Add the last line if it has content
        if current_line and len(lines) < max_lines:
            lines.append(current_line)
        
        return lines

    def _wrap_text_simple(self, text, font, max_width):
        """Simple text wrapping as fallback"""
        try:
            avg_char_width = font.getlength('A')
        except AttributeError:
            avg_char_width = font.size * 0.6
        
        chars_per_line = max(1, int(max_width / avg_char_width))
        return textwrap.wrap(text, width=chars_per_line, break_long_words=False, break_on_hyphens=False)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "PIP_TextToImage": PIP_TextToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PIP_TextToImage": "PIP 文本转图像",
}
