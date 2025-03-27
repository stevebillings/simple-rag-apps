import json
from typing import Dict


class Config:
    def __init__(self, config_file_path: str):
        with open(config_file_path, "r") as f:
            self.config_data = json.load(f)

    def get_bot_prompt(self) -> str:
        return self.config_data["bot_prompt"]

    def get_system_prompt_content_template(self) -> str:
        return self.config_data["system_prompt_content_template"]

    def get_vector_db_index_name(self) -> str:
        return self.config_data["vector_db_index_name"]

    def get_vector_db_namespace(self) -> str:
        return self.config_data["vector_db_namespace"]

    def get_faq(self) -> Dict[str, str]:
        return self.config_data["faq"]
