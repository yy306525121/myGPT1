import os
from typing import List, Any

from langchain import LlamaCpp, ConversationChain, PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory


class LlamaCppModel(object):
    def __init__(self, model_path: str = None, model_n_ctx: int = 1000, model_n_batch: int = 8,
                 callback: BaseCallbackHandler = None, verbose: bool = False, gpu_layers: int = 0):
        """
        通过LlamaCpp加载模型
        :param model_path: 模型地址
        :param model_n_ctx: 大模型的最大token限制，设置为4096（同llama.cpp -c参数）。16K长上下文版模型可适当调高，不超过16384（16K)
        :param model_n_batch: prompt批处理大小（同llama.cpp -b参数）
        :param callback:
        :param verbose:
        :param gpu_layers: 与llama.cpp中的-ngl参数一致，定义使用GPU的offload层数；苹果M系列芯片指定为1即可，不适用GPU的话设置为0即可
        """
        self._model = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch,
                               callback=[callback], verbose=verbose, n_gpu_layers=gpu_layers)

    def qa(self, retriever):
        return RetrievalQA.from_chain_type(llm=self._model, chain_type="stuff", retriever=retriever,
                                           return_source_documents=True)

    def new_chain(self, **kwargs: Any):
        human_prefix = kwargs.get('human_prefix', "Human")
        return ConversationChain(
            llm=self._model,
            prompt=self.get_prompt_template(human_prefix),
            callbacks=[self.handler],
            memory=ConversationBufferWindowMemory(
                k=3,
                human_prefix=human_prefix
            )
        )

    def get_prompt_template(self, human_prefix: str = 'User'):
        B_INST, E_INST = '[INST]', '[/INST]'
        B_SYS, E_SYS = '<<SYS>>\n', '\n<<SYS>>\n\n'
        instruction = '聊天记录:\n\n{history} \n\{human_prefix}：{input}'
        system_prompt = B_SYS + '你是一个可爱的人工智能机器人'

        template = B_INST + system_prompt + instruction + E_INST + '\nAI:'

        return PromptTemplate(
            template=template,
            input_variables=['history', 'input'],
            partial_variables={'human_prefix': human_prefix}
        )
