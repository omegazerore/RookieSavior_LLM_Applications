from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate


def build_standard_chat_prompt_template(kwargs):
    """Builds a standardized LangChain ChatPromptTemplate from input prompts.


    Supports both system and human prompts, including multimodal (text and image)
    configurations.
    
    
    Args:
    kwargs (dict): Dictionary containing optional `system` and `human` keys.
    - system (dict or list[dict]): System prompt configuration(s).
    - human (dict or list[dict]): Human prompt configuration(s).
    
    
    Returns:
    ChatPromptTemplate: A composed chat prompt template containing system and
    human message prompts.
    """
    
    messages = []

    if 'system' in kwargs:
        content = kwargs.get('system')

        # allow list of prompts for multimodal
        if isinstance(content, list):
            prompts = [PromptTemplate(**c) for c in content]
        else:
            prompts = [PromptTemplate(**content)]

        message = SystemMessagePromptTemplate(prompt=prompts)
        messages.append(message)

    if 'human' in kwargs:
        content = kwargs.get('human')

        # allow list of prompts for multimodal
        if isinstance(content, list):
            prompts = []
            for c in content:
                if c.get("type") == "image":
                    prompts.append(ImagePromptTemplate(**c))
                else:
                    prompts.append(PromptTemplate(**c))
        else:
            if content.get("type") == "image":
                prompts = [ImagePromptTemplate(**content)]
            else:
                prompts = [PromptTemplate(**content)]

        message = HumanMessagePromptTemplate(prompt=prompts)
        messages.append(message)

    chat_prompt_template = ChatPromptTemplate.from_messages(messages)
    
    return chat_prompt_template