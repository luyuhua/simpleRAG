import os 
from time import time
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.evaluation import FaithfulnessEvaluator,RelevancyEvaluator
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.base.llms.types import ChatMessage, MessageRole
# import nest_asyncio
# nest_asyncio.apply()

from loader import Loader
from db import DB



class Rag:
    db = DB()
    loader = Loader()

    def __init__(self) -> None:
        # Settings.llm = Ollama(model="qwen:32b",temperature=0.0, request_timeout=60.0)
        Settings.llm = Ollama(model="qwen:14b",temperature=0.0,request_timeout=60.0)
        Settings.embed_model = self.init_embedding_model()
        self.index = None
    
    def init_embedding_model(self):
        os.environ['https_proxy']='http://127.0.0.1:7890'
        model_name = "BAAI/bge-large-zh-v1.5"
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
        model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            query_instruction="为这个句子生成表示以用于检索相关文章："
        )
        model.query_instruction = "为这个句子生成表示以用于检索相关文章："
        embed_model = LangchainEmbedding(model)
        return embed_model
    
    ## LLM对话
    def ask(self,question,kb_names,history=[],stream=False):
        history_ = []
        for his in history:
            history_.append(ChatMessage(content=his[0],role=MessageRole.USER))
            history_.append(ChatMessage(content=his[1],role=MessageRole.ASSISTANT))
        # history = [ChatMessage(content=item,role=item) for item in history]
        print('init index start',time())
        indexs = [self.db.get_index(kb_name=kb_name) for kb_name in kb_names]
        print('init retriever start',time())
        if len(indexs) == 1:
            retriever = VectorIndexRetriever(indexs[0],similarity_top_k=3)
        else:
            retriever = QueryFusionRetriever(retrievers=[index.as_retriever() for index in indexs],
                                             similarity_top_k=3,
                                             num_queries=1,
                                             use_async=False)
        print('init query_engine start',time())
        # engine = RetrieverQueryEngine.from_args(retriever=retriever,streaming=stream)
        engine = ContextChatEngine.from_defaults(retriever=retriever)
        print('engine query start',time())
        # output = engine.query(question)  
        output = engine.stream_chat(question,chat_history=history_) if stream else engine.chat(question,chat_history=history_)
        print('engine query end',time())
        return output
    
    def eval(self,question):
        llm = Ollama(model="qwen:32b",temperature=0.0, request_timeout=60.0)
        response = self.ask(question)

        relevancyEvaluator = RelevancyEvaluator(llm=llm)
        faithfulnessEvaluator =FaithfulnessEvaluator(llm=llm)
        result1 = relevancyEvaluator.evaluate_response(query=question,response=response)
        result2 = faithfulnessEvaluator.evaluate_response(response=response)

    def create_questions(self,docs,question_num=1):
        llm = Ollama(model="qwen:32b",temperature=0.0, request_timeout=60.0)
        documents = self.loader.load_data(docs=docs)
        dataset_generator = RagDatasetGenerator.from_documents(
            documents=documents,
            llm=llm,
            num_questions_per_chunk=question_num,  # set the number of questions per nodes
        )

        rag_dataset = dataset_generator.generate_questions_from_nodes()
        # rag_dataset = dataset_generator.generate_dataset_from_nodes()
        questions = [e.query for e in rag_dataset.examples]
        print(questions)



if __name__ == "__main__":
    rag = Rag()
    kbs = rag.db.list_kb()
    rag.ask('什么是决策树',kb_names=[kbs[0].name,],stream=True)
