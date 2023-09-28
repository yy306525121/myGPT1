import copy
import os
from queue import Empty, Queue
from threading import Thread
from typing import Optional, Union, Any, Generator
from uuid import UUID

import gradio
from langchain import LlamaCpp, PromptTemplate, LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.schema.output import GenerationChunk, ChatGenerationChunk

llm = LlamaCpp(model_path=os.environ.get('model_path'),
               max_tokens=1024,
               streaming=True,
               top_p=0.1,
               top_k=40,
               temperature=0.15,
               repeat_penalty=1.1,
               n_gpu_layers=0)

system_message = """
    你是一个可爱的人工智能，你将帮助客户解答各种疑问.
"""


class LlamaCppCallback(BaseCallbackHandler):

    def __init__(self, queue: Queue = None):
        self.queue = queue

    def on_llm_new_token(self, token: str, *, chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None, run_id: UUID,
                         parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        print(token)
        self.queue.put(token)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        self.queue.empty()


def load_history(message, history):
    input_prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n "
    for interaction in history:
        input_prompt = input_prompt + str(interaction[0]) + " [/INST] " + str(interaction[1]) + " </s><s> [INST] "
    input_prompt = input_prompt + str(message) + " [/INST] "
    return input_prompt

q = Queue()
def predict(message, history):
    # message_history = RedisChatMessageHistory(session_id='yangzy_messages', url='redis://localhost:6379/0')
    # memory = ConversationBufferMemory(memory_key="chat_history",
    #                                   chat_memory=message_history,
    #                                   input_key="human_input", return_messages=True)

    # prompt = load_history(message, history)
    # output = llm(prompt=prompt)
    # temp = ""
    # for out in output:
    #     stream = copy.deepcopy(out)
    #     # temp += stream["choices"][0]["text"]
    #     temp += stream
    #     yield temp
    template = '我想去{location}旅行，我应该怎么做？'
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    # print(chain.predict(callbacks=[LlamaCppCallback()]))

    output = chain.run(location=str(message), callbacks=[LlamaCppCallback(queue=q)])
    temp = ''
    for out in output:
        stream = copy.deepcopy(out)
        # temp += stream["choices"][0]["text"]
        temp += stream
        yield temp

def stream(input_text) -> Generator:
    q = Queue()
    job_done = object()

    template = '我想去{location}旅行，我应该怎么做？'
    prompt = PromptTemplate.from_template(template)
    chain= LLMChain(llm=llm, prompt=prompt, verbose=True)

    def task():
        resp = chain.run(location=str(input_text), callbacks=[LlamaCppCallback(queue=q)])
        q.put(job_done)

    t = Thread(target=task)
    t.start()
    content = ""
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            content += next_token
            yield next_token, content
        except Empty:
            continue

def ask_llm(message, history):
    for next_token, content in stream(message):
        yield(content)


def gradio_ui():
    chatBot = gradio.Chatbot(value=[['你好', '你呀好'], ['你叫什么', '我叫小江']])
    return gradio.ChatInterface(fn=ask_llm, chatbot=chatBot).queue()
