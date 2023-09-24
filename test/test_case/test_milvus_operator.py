import os
from unittest import TestCase

from dotenv import load_dotenv
from langchain.document_loaders import JSONLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus

from app.core.config import settings
from app.db import MilvusOperator


class MilvusOperatorTest(TestCase):

    def setUp(self) -> None:
        if not load_dotenv(dotenv_path=settings.config_path / '.env'):
            print('加载.env文件失败，1：.env文件是否存。2：.env文件是否为空')
            exit(1)

    def test_add_documents(self):
        loader = JSONLoader(
            file_path='/Users/yangzy/Documents/content.json',
            jq_schema='.[].content')
        documents = loader.load()

        milvus = MilvusOperator()
        milvus.add_documents(documents)

    def test_search(self):
        milvus = MilvusOperator()
        result = milvus.search(content='李四')
        print(result)

