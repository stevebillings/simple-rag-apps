from typing import Dict
from src.config.config import Config

class ConfigFaq(Config):

    def get_bot_prompt(self):
        return "Ask a question about our e-Commerce store"
    
    def get_vector_db_index_name(self):
        return "faq-database"

    def get_vector_db_namespace(self):
        return "faq"
    
    def get_system_prompt_content_template(self):
        return """
            You are a helpful e-Commerce assistant helping customers with their general questions regarding policies and procedures when buying in our store.
            Our store sells e-books and courses for IT professionals. 

            Base your answers on the information in the following context. If the context does not contain the information, say that you don't know:

            Context: {}
        """
    
    def get_faq(self) -> Dict[str, str]:
        return {
            "How do I track my order?": "The best way to track your order is to log in, click on 'Account' in the top right corner of any page, select 'Orders' from the menu, select the order from the list, and click 'Track'.",
            "Cancelling an order": "Within 30 minutes after placing an order, you can cancel the order. To do this, log in, click on 'Account' in the top right corner of any page, select 'Orders' from the menu, select the order from the list, and click 'Cancel'.",
            "Placing an order": "To place an order, log in, select 'Products' from the menu, click the 'Buy' button next to the product you want, and then click 'Submit'.",
        }