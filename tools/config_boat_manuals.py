from tools.config import Config

class ConfigBoatManuals(Config):
    
    def get_vector_db_index_name(self):
        return "faq-database"

    def get_vector_db_namespace(self):
        return "boat-manuals"
    
    def get_system_prompt_content_template(self):
        return """
            You are a helpful assistant helping customers with their general questions about the boat described in the text from the boat's user manual given below. 

            Base your answers on the information in the given text from the boat user manual. \
            If the given text from the boat user manual does not contain the information, say that the user manual does not contain the information.
            If the given text from the boat user manual does contain information related to the question, provide as much information as you derive from
            the text from the boat user manual.

            Text from the boat user manual: {}
        """
    
    def get_faq(self):
        raise NotImplementedError("This method is not implemented for the boat manual application.")