import os
import io
import base64
from typing import List, Optional

import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from PIL import Image
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain_core.messages.human import HumanMessage
from unstructured.documents.elements import CompositeElement
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import chain, Runnable
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate

from src.initialization import credential_init


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

credential_init()

model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
           model_name="gpt-4o-mini-2024-07-18", temperature=0)


system_template = """
                  You are a helpful AI assistant as a personal assistant of a trend manager. You always do the best you can and pay attention to details.
                  
                  The trend manager analyzing multiple reports to identify subtrends within four megatrend areas:
                  1. Society: Related to consumer behavior, populations, or generational trends.
                  2. Technology: Includes AI, AR, beauty tech, and other emerging technologies.
                  3. Environment: Focuses on sustainability, ecological impact, and green innovation.
                  4. Industry: Covers movements in skincare, color cosmetics, personal care, and beauty innovation.
                  
                  For each subtrend, the trend manager needs:
                  •	A short definition or description of the trend.
                  •	Examples of products, brands, or innovations mentioned in the reports.
                  Please cluster the identified subtrends under the appropriate megatrend category, 
                  ensuring the analysis is concise and actionable. If specific examples are unavailable, highlight the general direction of the trend instead

                 """


def build_standard_chat_prompt_template(kwargs) -> Runnable:
    messages = []

    for key in ['system', 'messages', 'human']:
        if kwargs.get(key):
            if key == 'system':
                system_content = kwargs['system']
                system_prompt = PromptTemplate(**system_content)
                message = SystemMessagePromptTemplate(prompt=system_prompt)
            else:
                human_content = kwargs['human']
                human_prompt = PromptTemplate(**human_content)
                message = HumanMessagePromptTemplate(prompt=human_prompt)

            messages.append(message)

    chat_prompt = ChatPromptTemplate.from_messages(messages)

    return chat_prompt


def image_to_base64(image_path):

  with Image.open(image_path) as image:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_str = base64.b64encode(buffered.getvalue())
  return image_str.decode('utf-8')


def get_summary(element, model):

  str_type = str(type(element))

  if 'CompositeElement' in str_type:
    prompt = f"Summarize the following text:\n\n{element}\n\nSummary:"
  if 'Table' in str_type:
    prompt = f"Summarize the following table:\n\n{element}\n\nSummary:"

  response = model.invoke(prompt)

  return response.content


def get_image_summary(filename, model):

  image_str = image_to_base64(filename)

  human_message = HumanMessage(content=[{'type': 'text',
                        'text': 'What is in this image?'},
                        {'type': 'image_url',
                         'image_url': {
                          'url': f"data:image/png;base64,{image_str}"}
                        }])

  response = model.invoke([human_message])

  return response.content


def elements_2_vectorstore(elements: List[CompositeElement], filename: str,
                           fig_dir: Optional[str] = None,
                           ):
    documents = []

    for element in elements:

        str_type = str(type(element))

        summary = get_summary(element, model)

        if 'CompositeElement' in str_type:
            type_ = 'text'
        if 'Table' in str_type:
            type_ = 'table'

        documents.append(Document(page_content=summary, metadata={'type': type_, "filename": filename}))

    for image_file in os.listdir(fig_dir):
        image_path = f'{fig_dir}/{image_file}'

        summary = get_image_summary(image_path, model)

        documents.append(
            Document(page_content=summary, metadata={'type': 'image', 'image_source': f'{fig_dir}/{image_file}',
                                                     "filename": filename}))

    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)

    return vectorstore


def create_trinity_vectorstore(filename: str):
    """
    Create a vectorstore from a pdf file.
    """

    # remove the ext of the filename

    dir_ = filename.split('.')[0].split("/")[-1]

    fig_dir = f"{dir_}/figures"

    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    else:
        return None

    elements = partition_pdf(filename,
                             chunking_strategy='by_title',
                             infer_table_structure=True,
                             extract_image_block_types=['Image', 'Table'],
                             max_characters=4000,
                             new_after_n_chars=3800,
                             combine_text_under_n_chars=2000,
                             extract_image_block_output_dir=fig_dir,
                             strategy='hi_res')

    vectorstore = elements_2_vectorstore(elements, filename=filename.split("/")[-1], fig_dir=fig_dir)

    return vectorstore


if __name__ == "__main__":

    from src.io.path_definition import get_datafetch

    dir_ = os.path.join(get_datafetch(), "pdf_folder")

    vectorstore_file = os.path.join(get_datafetch(), "faiss_index")
    if os.path.isfile(vectorstore_file):
        vectorstore_main = FAISS.load_local(
        vectorstore_file, embeddings, allow_dangerous_deserialization=True
        )

    for filename in os.listdir(dir_):

        vectorstore = create_trinity_vectorstore(os.path.join(dir_, filename))

        if vectorstore is None:
            continue

        try:
            vectorstore_main.merge_from(vectorstore)
        except:
            vectorstore_main = vectorstore

    vectorstore_main.save_local(vectorstore_file)
