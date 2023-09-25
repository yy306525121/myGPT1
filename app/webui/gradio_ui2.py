import copy
import random
import time

import gradio
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import RedisChatMessageHistory, ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from llama_cpp import Llama

llm = Llama(
    model_path='/Users/yangzy/Documents/model/Llama2-chat-13B-Chinese-50W/ggml-model-q4_0.gguf',
    n_ctx=2048,
    verbose=False,
    n_gpu_layers=0,  # change n_gpu_layers if you have more or less VRAM
)


def generate_text(message, history):
    message_history = RedisChatMessageHistory(session_id='yangzy_session', url='redis://localhost:6379/0')
    template = """你是一个可爱的人工智能机器人.

    {chat_history}
    我: {human_input}
    机器人:"""
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template=template
    )
    memory = ConversationBufferMemory(memory_key='chat_memory', chat_memory=message_history, input_key='human_input',
                                      return_messages=True, k=15)
    chain = load_qa_chain(llm=llm, chain_type="stuff", memory=memory, prompt=prompt)
    replystr = chain({"human_input": message}, return_only_outputs=True)
    print(replystr)


system_message = """
    你是一个可爱的人工智能，你将帮助客户解答各种疑问.
"""


def load_history(message, history):
    input_prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n "
    for interaction in history:
        input_prompt = input_prompt + str(interaction[0]) + " [/INST] " + str(interaction[1]) + " </s><s> [INST] "
    input_prompt = input_prompt + str(message) + " [/INST] "
    return input_prompt


def predict(message, history):
    # message_history = RedisChatMessageHistory(session_id='yangzy_messages', url='redis://localhost:6379/0')
    # memory = ConversationBufferMemory(memory_key="chat_history",
    #                                   chat_memory=message_history,
    #                                   input_key="human_input", return_messages=True)

    prompt = load_history(message, history)
    output = llm(stream=True, prompt=prompt, temperature=0.15,
                 top_p=0.1,
                 top_k=40,
                 repeat_penalty=1.1,
                 max_tokens=1024, )
    temp = ""
    for out in output:
        stream = copy.deepcopy(out)
        temp += stream["choices"][0]["text"]
        yield temp


def gradio_ui():
    chatBot = gradio.Chatbot(value=[['你好', '你呀好'], ['你叫什么', '我叫小江']])
    return gradio.ChatInterface(fn=predict, chatbot=chatBot).queue()
