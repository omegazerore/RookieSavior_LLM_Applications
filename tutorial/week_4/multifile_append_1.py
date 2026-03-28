from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def build_standard_chat_prompt_template(kwargs):

    messages = []
 
    if 'system' in kwargs:
        content = kwargs.get('system')
        prompt = PromptTemplate(**content)
        message = SystemMessagePromptTemplate(prompt=prompt)
        messages.append(message)  

    if 'human' in kwargs:
        content = kwargs.get('human')
        prompt = PromptTemplate(**content)
        message = HumanMessagePromptTemplate(prompt=prompt)
        messages.append(message)
        
    chat_prompt = ChatPromptTemplate.from_messages(messages)
    
    return chat_prompt