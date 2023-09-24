import copy
import os
import time

import gradio
from langchain import LlamaCpp
from langchain.callbacks import StreamingStdOutCallbackHandler
from llama_cpp import Llama

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

llm = LlamaCpp(model_path='/Users/yangzy/Documents/model/Llama2-chat-13B-Chinese-50W/ggml-model-q4_0.gguf', max_tokens=1000,
                                n_gpu_layers=0)

llm = Llama(
    model_path='/Users/yangzy/Documents/model/Llama2-chat-13B-Chinese-50W/ggml-model-q4_0.gguf',
    n_ctx=2048,
    n_gpu_layers=0, # change n_gpu_layers if you have more or less VRAM
)
system_message = """
你是一个可爱的人工智能，你将帮助客户解答各种疑问.
"""

def generate_text(message, history):
    temp = ""
    input_prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n "
    for interaction in history:
        input_prompt = input_prompt + str(interaction[0]) + " [/INST] " + str(interaction[1]) + " </s><s> [INST] "

    input_prompt = input_prompt + str(message) + " [/INST] "

    output = llm(
        input_prompt,
        temperature=0.15,
        top_p=0.1,
        top_k=40,
        repeat_penalty=1.1,
        max_tokens=1024,
        stop=[
            "<|prompter|>",
            "<|endoftext|>",
            "<|endoftext|> \n",
            "ASSISTANT:",
            "USER:",
            "SYSTEM:",
        ],
        stream=True
    )
    for out in output:
        stream = copy.deepcopy(out)
        temp += stream["choices"][0]["text"]
        yield temp

    history = ["init", input_prompt]

def alternatingly_agree(message, history):
    if len(history) % 2 == 0:
        return f"Yes, I do think that '{message}'"
    else:
        return "I don't think so"


def predict(message, history):
    for i in range(len(message)):
        time.sleep(0.3)
        yield "You typed: " + message[: i + 1]

def gradio_ui():
    # io = gradio.Interface(fn=generate_chat, inputs=gradio.components.Textbox(label='问题'),
    #                       outputs=gradio.components.Textbox(label='答案'), )
    return gradio.ChatInterface(generate_text).queue()
