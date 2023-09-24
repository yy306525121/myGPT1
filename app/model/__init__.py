import sys
from abc import ABCMeta
from typing import List, Optional, Union, Any, Dict
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.output import GenerationChunk, ChatGenerationChunk


class StreamingWebCallbackHandler(BaseCallbackHandler):
    tokens: List[str] = []
    is_responding: bool = False
    response_id: str
    response: str = None

    def on_llm_new_token(self, token: str, *, chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
                         run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        sys.stdout.write(token)
        sys.stdout.flush()
        print('aaaaa')
        self.tokens.append(token)

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID,
                       parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        self.is_responding = True
        self.response_id = run_id
        self.response = None


