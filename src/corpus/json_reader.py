import json
import os
from typing import Any, Dict, List


# TODO: So far this is identical to FaqReader. Eliminate faq reader if that remains true
class JsonReader:
    def __init__(self, corpus_dir_path: str, workspace_root: str) -> None:
        self._corpus_dir_path = corpus_dir_path
        self._workspace_root = workspace_root

    def read_json_list(self) -> List[Dict[str, Any]]:
        if not os.path.isabs(self._corpus_dir_path):
            corpus_dir = os.path.join(self._workspace_root, self._corpus_dir_path)
        else:
            corpus_dir = self._corpus_dir_path

        combined_data: List[Dict[str, Any]] = []
        for filename in os.listdir(corpus_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(corpus_dir, filename)
                with open(file_path, "r") as f:
                    file_data = json.load(f)
                    # Merge the data, assuming each JSON file contains a dictionary
                    combined_data.extend(file_data)

        return combined_data
