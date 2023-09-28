import copy
import os

import gradio
from llama_cpp import Llama

llm = Llama(
    model_path=os.environ.get('model_path'),
    n_ctx=2048,
    verbose=False,
    n_gpu_layers=0,  # change n_gpu_layers if you have more or less VRAM
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


def gradio_ui():
    return gradio.ChatInterface(generate_text).queue()
