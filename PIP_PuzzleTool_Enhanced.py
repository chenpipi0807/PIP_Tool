import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import requests
from io import BytesIO
from pathlib import Path
import re

class PIP_PuzzleTool_Enhanced:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),  # Accept IMAGE type (may contain tensor or URL string)
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
                # Convert the input to tensor (handles both tensor and URL inputs)
                img_tensor = self._convert_input_to_tensor(img)
                if img_tensor is not None:
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
                draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))
            
            x_offset += img.width + inter_image_margin
        
        # Generate description
        titles = [string1, string2, string3, string4, string5]
        description = self._generate_description(titles)
        
        # Convert result back to tensor
        result_tensor = self._pil_to_tensor(result).unsqueeze(0)  # Add batch dimension
        
        return (result_tensor, description)

    def _convert_input_to_tensor(self, input_data):
        """Convert input to tensor, handling both IMAGE tensors and URL strings wrapped in IMAGE type"""
        if input_data is None:
            return None
            
        print(f"Debug: Input data type: {type(input_data)}")
        print(f"Debug: Input data: {input_data}")
        
        # Check if it's already a tensor (traditional IMAGE input)
        if isinstance(input_data, torch.Tensor):
            print("Debug: Processing as tensor input")
            print(f"Debug: Original tensor shape: {input_data.shape}")
            # Ensure image is the correct dimension (batch_size, height, width, channels)
            if input_data.dim() == 3:
                input_data = input_data.unsqueeze(0)
            # Only take the first image from each batch
            tensor = input_data[0]
            
            # Handle RGBA to RGB conversion if needed
            if tensor.shape[-1] == 4:  # RGBA format
                print("Debug: Converting RGBA to RGB by removing alpha channel")
                # Take only RGB channels, ignore alpha
                tensor = tensor[:, :, :3]
            elif tensor.shape[-1] != 3:
                print(f"Warning: Unexpected number of channels: {tensor.shape[-1]}")
            
            print(f"Debug: Final tensor shape: {tensor.shape}")
            return tensor
        
        # Check if it's a string (URL input wrapped in IMAGE type)
        elif isinstance(input_data, str):
            print(f"Debug: Processing as string input: {input_data}")
            # Check if it looks like a URL
            if self._is_url(input_data):
                return self._download_image_from_url(input_data)
            else:
                print(f"Warning: String input '{input_data}' doesn't appear to be a valid URL")
                return None
        
        # Check if it's a list/tuple (sometimes ComfyUI passes data in lists)
        elif isinstance(input_data, (list, tuple)) and len(input_data) > 0:
            print(f"Debug: Processing as list/tuple input, length: {len(input_data)}")
            # Try first element
            first_element = input_data[0]
            if isinstance(first_element, str) and self._is_url(first_element):
                return self._download_image_from_url(first_element)
            else:
                return self._convert_input_to_tensor(first_element)
        
        else:
            print(f"Warning: Unsupported input type: {type(input_data)}")
            return None

    def _is_url(self, string):
        """Check if a string looks like a URL"""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(string) is not None

    def _download_image_from_url(self, url):
        """Download image from URL and convert to tensor"""
        try:
            print(f"Downloading image from URL: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Convert response content to BytesIO
            image_bytesio = BytesIO(response.content)
            
            # Open image with PIL
            image = Image.open(image_bytesio)
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to tensor
            image_array = np.array(image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(image_array)
            
            print(f"Successfully downloaded and converted image. Shape: {tensor.shape}")
            return tensor
            
        except Exception as e:
            print(f"Error downloading image from URL {url}: {e}")
            return None

    def _find_font_path(self, font_filename):
        """Search for the font file in various possible locations"""
        # Search in common directories
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


class PIP_DynamicPuzzleSelection_Enhanced:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),  # Accept IMAGE type (may contain tensor or URL string)
                "string1": ("STRING", {"default": ""}),
                "selection_string": ("STRING", {"default": "1,2,3"}),
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
                "use_english_description": ("BOOLEAN", {"default": False}),
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

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "clean_image", "description")
    FUNCTION = "select_and_stitch"
    CATEGORY = "图像处理"

    def select_and_stitch(self, image1, string1, selection_string, inter_image_margin, image_height, 
                         bottom_margin, font_size, use_english_description,
                         image2=None, string2="", image3=None, string3="", 
                         image4=None, string4="", image5=None, string5=""):
        # Parse selection string (e.g., "1,3,5" means use images 1, 3, and 5)
        try:
            selections = [int(x.strip()) for x in selection_string.split(',') if x.strip().isdigit()]
            selections = [x for x in selections if 1 <= x <= 5]  # Ensure valid range
        except:
            print(f"Error parsing selection string: {selection_string}")
            selections = [1]  # Default to just the first image

        # Find and load the font
        font_path = self._find_font_path("Arial_Unicode.ttf")
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"Error loading font: {e}")
            font = ImageFont.load_default()

        # Collect all images and titles
        all_images = [
            (image1, string1), (image2, string2), (image3, string3), 
            (image4, string4), (image5, string5)
        ]
        
        # Convert inputs to tensors and filter based on selection
        images_data = []
        titles = []
        for sel in selections:
            if sel <= len(all_images):
                img, text = all_images[sel-1]  # Convert to 0-based index
                if img is not None:
                    # Convert the input to tensor (handles both tensor and URL inputs)
                    img_tensor = self._convert_input_to_tensor(img)
                    if img_tensor is not None:
                        images_data.append((img_tensor, text))
                        titles.append(text)
        
        if not images_data:
            print("No valid images found!")
            # Return empty black image
            empty_img = torch.zeros((1, image_height, 100, 3))
            return (empty_img, empty_img, "")
        
        # Convert tensors to PIL images and resize
        pil_images = []
        for img_tensor, text in images_data:
            pil_img = self._tensor_to_pil(img_tensor)
            # Resize while maintaining aspect ratio
            aspect_ratio = pil_img.width / pil_img.height
            new_width = int(image_height * aspect_ratio)
            resized_img = pil_img.resize((new_width, image_height), Image.LANCZOS)
            pil_images.append((resized_img, text))

        # Calculate the total width
        total_width = sum(img.width for img, _ in pil_images) + inter_image_margin * (len(pil_images) - 1)
        
        # Create clean image (without text)
        clean_result = Image.new('RGB', (total_width, image_height), (0, 0, 0))
        
        # Create image with text
        max_text_height = font_size * 2
        text_result_height = image_height + max_text_height + bottom_margin
        text_result = Image.new('RGB', (total_width, text_result_height), (0, 0, 0))
        draw = ImageDraw.Draw(text_result)
        
        # Paste images
        x_offset = 0
        for img, text in pil_images:
            # Paste to both results
            clean_result.paste(img, (x_offset, 0))
            text_result.paste(img, (x_offset, 0))
            
            # Draw text only on text_result
            if text:
                try:
                    text_width = draw.textlength(text, font=font)
                except AttributeError:
                    text_width = font.getlength(text)
                
                text_x = x_offset + (img.width - text_width) // 2
                text_y = image_height + 10
                draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))
            
            x_offset += img.width + inter_image_margin
        
        # Generate description
        if use_english_description:
            description = self._generate_english_description(titles)
        else:
            description = self._generate_chinese_description(titles)
        
        # Convert results back to tensors
        text_result_tensor = self._pil_to_tensor(text_result).unsqueeze(0)
        clean_result_tensor = self._pil_to_tensor(clean_result).unsqueeze(0)
        
        return (text_result_tensor, clean_result_tensor, description)

    # Copy all helper methods from the enhanced puzzle tool
    def _convert_input_to_tensor(self, input_data):
        """Convert input to tensor, handling both IMAGE tensors and URL strings wrapped in IMAGE type"""
        if input_data is None:
            return None
            
        print(f"Debug: Input data type: {type(input_data)}")
        print(f"Debug: Input data: {input_data}")
        
        # Check if it's already a tensor (traditional IMAGE input)
        if isinstance(input_data, torch.Tensor):
            print("Debug: Processing as tensor input")
            print(f"Debug: Original tensor shape: {input_data.shape}")
            # Ensure image is the correct dimension (batch_size, height, width, channels)
            if input_data.dim() == 3:
                input_data = input_data.unsqueeze(0)
            # Only take the first image from each batch
            tensor = input_data[0]
            
            # Handle RGBA to RGB conversion if needed
            if tensor.shape[-1] == 4:  # RGBA format
                print("Debug: Converting RGBA to RGB by removing alpha channel")
                # Take only RGB channels, ignore alpha
                tensor = tensor[:, :, :3]
            elif tensor.shape[-1] != 3:
                print(f"Warning: Unexpected number of channels: {tensor.shape[-1]}")
            
            print(f"Debug: Final tensor shape: {tensor.shape}")
            return tensor
        
        # Check if it's a string (URL input wrapped in IMAGE type)  
        elif isinstance(input_data, str):
            print(f"Debug: Processing as string input: {input_data}")
            # Check if it looks like a URL
            if self._is_url(input_data):
                return self._download_image_from_url(input_data)
            else:
                print(f"Warning: String input '{input_data}' doesn't appear to be a valid URL")
                return None
        
        # Check if it's a list/tuple (sometimes ComfyUI passes data in lists)
        elif isinstance(input_data, (list, tuple)) and len(input_data) > 0:
            print(f"Debug: Processing as list/tuple input, length: {len(input_data)}")
            # Try first element
            first_element = input_data[0]
            if isinstance(first_element, str) and self._is_url(first_element):
                return self._download_image_from_url(first_element)
            else:
                return self._convert_input_to_tensor(first_element)
        
        else:
            print(f"Warning: Unsupported input type: {type(input_data)}")
            return None

    def _is_url(self, string):
        """Check if a string looks like a URL"""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(string) is not None

    def _download_image_from_url(self, url):
        """Download image from URL and convert to tensor"""
        try:
            print(f"Downloading image from URL: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Convert response content to BytesIO
            image_bytesio = BytesIO(response.content)
            
            # Open image with PIL
            image = Image.open(image_bytesio)
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to tensor
            image_array = np.array(image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(image_array)
            
            print(f"Successfully downloaded and converted image. Shape: {tensor.shape}")
            return tensor
            
        except Exception as e:
            print(f"Error downloading image from URL {url}: {e}")
            return None

    def _find_font_path(self, font_filename):
        """Search for the font file in various possible locations"""
        # Search in common directories
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
        
    def _generate_chinese_description(self, titles):
        """Generate a Chinese description string for the images from left to right"""
        # Filter out empty titles
        valid_titles = [title for title in titles if title.strip()]
        
        if not valid_titles:
            return ""
            
        # Format: 这张图片从左到右分别是title1|title2|title3|...
        title_part = "|".join(valid_titles)
        description = f"这张图片从左到右依次是{title_part}"
        
        return description

    def _generate_english_description(self, titles):
        """Generate an English description string for the images from left to right"""
        # Filter out empty titles
        valid_titles = [title for title in titles if title.strip()]
        
        if not valid_titles:
            return ""
            
        # Format: This image shows title1|title2|title3|... from left to right
        title_part = "|".join(valid_titles)
        description = f"This image shows {title_part} from left to right"
        
        return description

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "PIP_PuzzleTool_Enhanced": PIP_PuzzleTool_Enhanced,
    "PIP_DynamicPuzzleSelection_Enhanced": PIP_DynamicPuzzleSelection_Enhanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PIP_PuzzleTool_Enhanced": "PIP 拼图工具 (增强版支持URL)",
    "PIP_DynamicPuzzleSelection_Enhanced": "PIP 拼图动态选择 (增强版支持URL)",
}
