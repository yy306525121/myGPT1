import os
from unittest import TestCase

from dotenv import load_dotenv
from langchain import LlamaCpp, LLMChain, PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

from app.core.config import settings
from app.db import MilvusOperator
from app.model import StreamingWebCallbackHandler
from app.model.llama_cpp_model import LlamaCppModel


class LlamaCppModelTest(TestCase):

    def setUp(self) -> None:
        if not load_dotenv(dotenv_path=settings.config_path / '.env'):
            print('加载.env文件失败，1：.env文件是否存。2：.env文件是否为空')
            exit(1)

    def test_qa(self):
        milvus = MilvusOperator(embedding_model_path=os.environ.get('embedding_model_path'))

        callback = StreamingWebCallbackHandler()
        model = LlamaCppModel(model_path=os.environ.get('model_path'), callback=callback)
        qa = model.qa(retriever=milvus.as_retriever())
        print(qa('介绍一下张三'))

    def test_chain(self):
        callback = StreamingWebCallbackHandler()

        llm = LlamaCpp(model_path='/Users/yangzy/Documents/model/Llama2-chat-13B-Chinese-50W/ggml-model-q4_0.gguf', max_tokens=1000, n_batch=8,
                 callback=[callback], verbose=True, n_gpu_layers=0)

        prompt = PromptTemplate(
            input_variables=["product"],
            template="给我讲个{product} 笑话",
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        print(chain.run("好笑的"))
