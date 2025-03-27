import json
import os
from typing import Dict


class FaqReader:
    def __init__(self, corpus_dir_path: str, workspace_root: str) -> None:
        self._corpus_dir_path = corpus_dir_path
        self._workspace_root = workspace_root

    def read_faq(self) -> Dict[str, str]:
        if not os.path.isabs(self._corpus_dir_path):
            corpus_dir = os.path.join(self._workspace_root, self._corpus_dir_path)
        else:
            corpus_dir = self._corpus_dir_path

        faq_file = os.path.join(corpus_dir, "faq.json")
        with open(faq_file, "r") as f:
            return json.load(f)
