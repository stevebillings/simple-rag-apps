import abc

class Config(abc.ABC):

    @abc.abstractmethod
    def get_bot_prompt(self):
        pass

    @abc.abstractmethod
    def get_system_prompt_content_template(self):
        pass

    @abc.abstractmethod
    def get_vector_db_index_name(self):
        pass

    @abc.abstractmethod
    def get_vector_db_namespace(self):
        pass

    @abc.abstractmethod
    def get_faq(self):
        pass