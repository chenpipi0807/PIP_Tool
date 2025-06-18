import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from pathlib import Path

class PIP_PuzzleTool:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "string1": ("STRING", {"default": ""}),
                "inter_image_margin": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 200,
                    "step": 1
                }),
                "image_height": ("INT", {
                    "default": 996,
                    "min": 100,
                    "max": 4096,
                    "step": 1
                }),
                "bottom_margin": ("INT", {
                    "default": 200,
                    "min": 0,
                    "max": 1000,
                    "step": 1
                }),
                "font_size": ("INT", {
                    "default": 40,
                    "min": 10,
                    "max": 200,
                    "step": 1
                }),
            },
            "optional": {
                "image2": ("IMAGE",),
                "string2": ("STRING", {"default": ""}),
                "image3": ("IMAGE",),
                "string3": ("STRING", {"default": ""}),
                "image4": ("IMAGE",),
                "string4": ("STRING", {"default": ""}),
                "image5": ("IMAGE",),
                "string5": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "description")
    FUNCTION = "stitch_images_with_text"
    CATEGORY = "图像处理"

    def stitch_images_with_text(self, image1, string1, inter_image_margin, image_height, bottom_margin, font_size, 
                                image2=None, string2="", image3=None, string3="", image4=None, string4="", image5=None, string5=""):
        # Find and load the font
        font_path = self._find_font_path("Arial_Unicode.ttf")
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"Error loading font: {e}")
            # Fall back to default font if Arial_Unicode.ttf is not available
            font = ImageFont.load_default()

        # Collect all provided images and their corresponding titles
        images_data = []
        for idx, (img, text) in enumerate([
            (image1, string1), (image2, string2), (image3, string3), 
            (image4, string4), (image5, string5)
        ], 1):
            if img is not None:
                # Ensure image is the correct dimension (batch_size, height, width, channels)
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                # Only take the first image from each batch
                img_tensor = img[0]
                images_data.append((img_tensor, text))
        
        # Convert tensors to PIL images and resize
        pil_images = []
        for img_tensor, text in images_data:
            pil_img = self._tensor_to_pil(img_tensor)
            # Resize while maintaining aspect ratio
            aspect_ratio = pil_img.width / pil_img.height
            new_width = int(image_height * aspect_ratio)
            resized_img = pil_img.resize((new_width, image_height), Image.LANCZOS)
            pil_images.append((resized_img, text))

        # Calculate the total width of the stitched image
        total_width = sum(img.width for img, _ in pil_images) + inter_image_margin * (len(pil_images) - 1)
        
        # Create blank image with black background
        # Calculate text height (approximately)
        max_text_height = font_size * 2  # Give some extra space for text
        result_height = image_height + max_text_height + bottom_margin
        result = Image.new('RGB', (total_width, result_height), (0, 0, 0))
        draw = ImageDraw.Draw(result)
        
        # Paste images and draw texts
        x_offset = 0
        for img, text in pil_images:
            # Paste image
            result.paste(img, (x_offset, 0))
            
            # Draw text centered below image
            if text:
                try:
                    # Get text size to center it
                    text_width = draw.textlength(text, font=font)
                except AttributeError:
                    # Fallback for older PIL versions
                    text_width = font.getlength(text)
                
                text_x = x_offset + (img.width - text_width) // 2
                text_y = image_height + 10  # Small gap between image and text
                
                # Draw the text with white color
                draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
            
            # Move x_offset for next image
            x_offset += img.width + inter_image_margin
        
        # Convert back to PyTorch tensor and add batch dimension
        result_tensor = self._pil_to_tensor(result)
        result_tensor = result_tensor.unsqueeze(0)
        
        # Generate descriptive string
        description = self._generate_description([text for _, text in pil_images])
        
        return (result_tensor, description)

    def _find_font_path(self, font_filename):
        """Search for the font file in various possible locations"""
        # List of possible font directories to search
        possible_dirs = [
            Path("fonts"),  # Relative to current directory
            Path(__file__).parent / "fonts",  # In the same directory as this file
            Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'fonts'))),  # ComfyUI root/fonts
            Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fonts'))),  # custom_nodes/fonts
            Path("C:/Windows/Fonts"),  # Windows system fonts
        ]
        
        # Search for the font file
        for dir_path in possible_dirs:
            if not dir_path.exists():
                continue
            
            font_path = dir_path / font_filename
            if font_path.exists():
                return str(font_path)
        
        # Return default if font not found
        print(f"Warning: Font {font_filename} not found. Using default font.")
        return font_filename  # Let PIL handle the error and fall back to default

    def _tensor_to_pil(self, tensor):
        """Convert a PyTorch tensor to a PIL Image"""
        if tensor is None:
            return None
        # Ensure tensor is in range 0-1
        tensor = tensor.clamp(0, 1)
        # Convert to numpy array and adjust to 0-255
        img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        # Create PIL image
        return Image.fromarray(img_np, mode='RGB')
    
    def _pil_to_tensor(self, pil_img):
        """Convert a PIL Image to a PyTorch tensor"""
        if pil_img is None:
            return None
        
        # Make sure the image is in RGB mode
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # Convert to numpy array
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        # Convert to PyTorch tensor
        return torch.from_numpy(img_np)
        
    def _generate_description(self, titles):
        """Generate a description string for the images from left to right"""
        # Filter out empty titles
        valid_titles = [title for title in titles if title.strip()]
        
        if not valid_titles:
            return ""
            
        # Format: 这张图片从左到右分别是title1|title2|title3|...
        title_part = "|".join(valid_titles)
        description = f"这张图片从左到右依次是{title_part}"
        
        return description
