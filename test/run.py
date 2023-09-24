import unittest

from test.test_case.test_llama_cpp_model import LlamaCppModelTest
from test.test_case.test_milvus_operator import MilvusOperatorTest

if __name__ == '__main__':
    suite = unittest.TestSuite()

    # suite.addTest(MilvusOperatorTest('test_add_documents'))
    # suite.addTest(MilvusOperatorTest('test_search'))
    suite.addTest(LlamaCppModelTest('test_qa'))

    # 运行测试
    runner = unittest.TextTestRunner()
    runner.run(suite)