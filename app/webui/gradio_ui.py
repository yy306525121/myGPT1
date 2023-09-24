import os
import time
from random import random

import gradio
from langchain import ConversationChain, LLMChain, LlamaCpp
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.schema import HumanMessage, AIMessage

from app.db import MilvusOperator
from app.model.llama_cpp_model import LlamaCppModel


def image_classifier(inp):
    return {'cat': 0.3, 'dog': 0.7}





def generate_chat(query: str = None):
    milvus = MilvusOperator(embedding_model_path=os.environ.get('embedding_model_path'))
    callback = [StreamingStdOutCallbackHandler()]
    model = LlamaCppModel(model_path=os.environ.get('model_path'), callback=callback)
    retriever = milvus.as_retriever()
    qa = model.qa(retriever=retriever)
    result = qa(query)
    return result['result']

def alternatingly_agree(message, history):
    if len(history) % 2 == 0:
        return f"Yes, I do think that '{message}'"
    else:
        return "I don't think so"

def slow_echo(message, history):
    for i in range(len(message)):
        time.sleep(0.3)
        yield "You typed: " + message[: i+1]

# def predict(message, history):
#     history_langchain_format = []
#     for human, ai in history:
#         history_langchain_format.append(HumanMessage(content=human))
#         history_langchain_format.append(AIMessage(content=ai))
#     history_langchain_format.append(HumanMessage(content=message))
#     gpt_response = llm(history_langchain_format)
#     return gpt_response.content

# def predict(message, history):
#     callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#     llm = LlamaCpp(
#         model_path="/Users/yangzy/Documents/model/Llama2-chat-13B-Chinese-50W/ggml-model-q4_0.gguf", callback_manager=callback_manager, verbose=True
#     )
#
#     prompt = """
#     Question: 你好
#     """
#     llm_chain = LLMChain(prompt=prompt, llm =llm)


def gradio_ui():
    # io = gradio.Interface(fn=generate_chat, inputs=gradio.components.Textbox(label='问题'),
    #                       outputs=gradio.components.Textbox(label='答案'), )
    return gradio.ChatInterface(slow_echo).queue()