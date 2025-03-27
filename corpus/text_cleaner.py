import re


class TextCleaner:

    def clean(self, text_to_clean: str) -> str:
        cleaned_text: str = re.sub(r"[^\w\s]", "", text_to_clean)
        return cleaned_text