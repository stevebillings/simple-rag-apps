from typing import Dict, Tuple


class FrequentlyAskedQuestions:

    def __init__(self) -> None:
        self._faq: Dict[str, str] = {
            "How do I track my order?": "The best way to track your order is to log in, click on 'Account' in the top right corner of any page, select 'Orders' from the menu, select the order from the list, and click 'Track'.",
            "Cancelling an order": "Within 30 minutes after placing an order, you can cancel the order. To do this, log in, click on 'Account' in the top right corner of any page, select 'Orders' from the menu, select the order from the list, and click 'Cancel'.",
        }

    def lookup_answer(self, question: str) -> str:
        return self._faq[question]
    
    def enumerate(self) -> enumerate[Tuple[str, str]]:
        return enumerate(self._faq.items())
