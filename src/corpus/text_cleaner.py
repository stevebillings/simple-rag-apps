import re


class TextCleaner:

    def clean(self, text_to_clean: str) -> str:
        # Keep word characters, whitespace, and the specified special characters
        cleaned_text: str = re.sub(r"[^\w\s.,!@#$%&*()+=:;'\"?/<>]", "", text_to_clean)
        return cleaned_text
