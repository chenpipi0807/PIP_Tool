import re

class PIP_SearchReplace:
    """
    A ComfyUI node for searching and replacing text with advanced options.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"multiline": True}),
                "search_text": ("STRING", {"multiline": False}),
                "replace_with": ("STRING", {"multiline": False}),
                "remove_special_chars": ("BOOLEAN", {"default": False, "label": "一键移除特殊符号"}),
                "remove_chinese_chars": ("BOOLEAN", {"default": False, "label": "一键移除中文"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "search_replace"
    CATEGORY = "PIP"
    
    def search_replace(self, string, search_text, replace_with, remove_special_chars=False, remove_chinese_chars=False):
        """
        Performs search and replace operations on the input string based on the specified parameters.
        
        Rules:
        1. Can replace with empty space or empty string
        2. Converts English text to lowercase for search, then restores original case
        3. Supports regex for special characters
        4. Special character removal excludes commas and periods (both English and Chinese)
        5. Can remove Chinese characters and punctuation
        """
        # Create a working copy of the string
        result = string
        
        # Step 1: Handle basic search and replace if search_text is provided
        if search_text:
            # For case-insensitive search with case preservation, we need a more complex approach
            # We'll use regex with case-insensitive flag for proper replacement of all occurrences
            if search_text.lower() != search_text.upper():  # If search text contains letters
                # Create a regex pattern that matches the search_text case-insensitively
                pattern = re.compile(re.escape(search_text), re.IGNORECASE)
                
                # Function to preserve case when replacing
                def replace_match(match):
                    matched_text = match.group(0)
                    # If replacement is empty, just return it
                    if not replace_with:
                        return replace_with
                    
                    # Try to maintain case pattern of the matched text in the replacement
                    if matched_text.isupper():
                        return replace_with.upper()
                    elif matched_text[0].isupper() and matched_text[1:].islower():
                        # Title case
                        return replace_with[0].upper() + replace_with[1:].lower() if len(replace_with) > 1 else replace_with.upper()
                    else:
                        return replace_with.lower()
                
                # Perform replacement on all occurrences while preserving case
                result = pattern.sub(replace_match, result)
            else:
                # For non-letter search terms (numbers, symbols), just do a regular replacement of all occurrences
                result = result.replace(search_text, replace_with)
        
        # Step 2: Handle special character removal if enabled
        if remove_special_chars:
            # Define allowed punctuation (commas and periods in both English and Chinese)
            allowed_punctuation = [',', '.', '，', '。']
            
            # Create a regex pattern to match special characters excluding allowed punctuation
            special_chars_pattern = r'[^\w\s' + re.escape(''.join(allowed_punctuation)) + ']'
            
            # Remove special characters
            result = re.sub(special_chars_pattern, '', result)
        
        # Step 3: Handle Chinese character removal if enabled
        if remove_chinese_chars:
            # This pattern matches Chinese characters and Chinese punctuation
            chinese_pattern = r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]'
            
            # Remove Chinese characters and punctuation
            result = re.sub(chinese_pattern, '', result)
        
        return (result,)
