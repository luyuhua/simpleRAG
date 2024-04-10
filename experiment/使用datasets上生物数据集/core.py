# Retrieval-Augmented Generation (RAG) System 

# Import Libraries
# ----------------
import os
from langchain_community.document_loaders import CSVLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import pandas as pd
import numpy as np
import json
import sys
sys.path.append(os.getcwd())
from models import OpenaiApi



EXP_NAME = 'base'


class Knowage:
    # Constants and API Keys
    # ----------------------
    DOCUMENTS_PATH = "docs"
    VECTOR_DB_DIRECTORY = "VectorStore"
    MEMORY_LEN= 5
    QA_CHAIN_PROMPT = PromptTemplate.from_template("""根据下面的上下文（context）内容回答问题。
    如果你不知道答案，就回答不知道，不要试图编造答案。
    答案最多3句话，保持答案简介。
    答案使用的语言与（问题）使用的语言保持一致。
    {context}
    问题：{question}
    """)

    JUDGY_PROMPT= PromptTemplate.from_template("""我将给你一个问题和一个对应的答案，这是一个答题者回答的，请对这个答题者的回答正确与否，与回答质量给出打分。  
    问题：{question}  
    标准答案：{gt_answer}  
    答案：{llm_answer}  
    以上是所有的问题和答案，请给该答题者的回答打分，满分 10 分。
    请直接输出你的分数值(阿拉伯数字形式)，不需要其他文字和符号。                                           
    """)

    def __init__(self):
        self.embeddings = self.init_embeddings()
        self.db = self.init_vector_database()
        self.retriever=self.db.as_retriever()
        self.LLM = self.initialize_chat_model()
        self.retrieval_qa_chain = self.init_retrieval_qa_chain()
        self.judgy_chain = LLMChain(llm=self.LLM,prompt=self.JUDGY_PROMPT)

    def init_embeddings(self):
        """Creates embeddings from text."""
        # model_name = "infgrad/stella-base-zh-v3-1792d"
        model_name = "/home/lu/.cache/torch/sentence_transformers/infgrad_stella-base-zh-v3-1792d"

        # model_kwargs = {'device': 'cuda:0'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    # model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs)
        print('init embedding finished')
        return embeddings

    def init_vector_database(self):
        db = Chroma(collection_name=EXP_NAME,embedding_function=self.embeddings, persist_directory=self.VECTOR_DB_DIRECTORY)
        if db._collection.count()==0:
            loader = CSVLoader(file_path='experiment/mini_passages.csv', source_column='id')
            docs = loader.load()
            db.add_documents(docs)
        return db

        # return Chroma.from_documents(documents=docs,embedding=self.embeddings,collection_name=EXP_NAME,persist_directory=self.VECTOR_DB_DIRECTORY)

        # return Chroma(collection_name=EXP_NAME,embedding_function=self.embeddings, persist_directory=self.VECTOR_DB_DIRECTORY)

    
    def initialize_chat_model(self):
        """Initializes the chat model with specified AI model."""
        # return OpenaiApi()
        return ChatOllama(model="qwen:14b")

    def init_retrieval_qa_chain(self):
        """Creates a retrieval QA chain combining model and database."""
        # memory = ConversationBufferWindowMemory(memory_key='chat_history', k=self.MEMORY_LEN, return_messages=True)
        # 要返回source文档，除了加 return_source_documents=True 外，还要在memory处指定 input 和 output 的 key，不然报错
        memory = ConversationBufferWindowMemory(memory_key='chat_history', k=self.MEMORY_LEN, input_key='question', output_key='answer', return_messages=True)
        return ConversationalRetrievalChain.from_llm(self.LLM, retriever=self.retriever, memory=memory,
                                                    combine_docs_chain_kwargs={'prompt': self.QA_CHAIN_PROMPT},
                                                    return_source_documents=True,)

    def eval_run(self):
        df = pd.read_csv('experiment/mini_qas.csv',index_col='id')
        df[['retriever_ids','llm_answer']] = df.apply(self.apply_chain_run,axis=1,result_type='expand')
        df.to_csv('experiment/result_retriever_LLManswer.csv')

    def eval_cal(self):
        df = pd.read_csv('experiment/result_retriever_LLManswer.csv',index_col='id')
        df_score = df.apply(self.apply_cal_retriever_score,axis=1,result_type='expand')
        mean_score = df_score.mean()
        print(f'{"recall":<10}: {mean_score[0]:.2f}\n{"precision":<10}: {mean_score[1]:.2f}\n{"f_score":<10}: {mean_score[2]:.2f}')
        answer_scores = df.apply(self.apply_judgy,axis=1)
        print(f'{"LLM回答质量分":<10}: {answer_scores.mean():.2f}')

        df[['recall','precision','f_score']] = df_score
        df['LLM回答质量分'] = answer_scores
        df.to_csv('experiment/result_score.csv')


    def apply_chain_run(self,x):
        question = x['question']
        retriever = self.db.as_retriever()
        docs = retriever.invoke(question)
        recall_ids = [int(doc.metadata['source']) for doc in docs]

        gt_ids = json.loads(x['relevant_passage_ids']) 
        gt_docs = []
        for gt_id in gt_ids:
            db_result = self.db.get(where={"source": str(gt_id)})
            gt_docs.append(Document(page_content=db_result['documents'][0],metadata=db_result['metadatas'][0]))

        document_chain = create_stuff_documents_chain(self.LLM, self.QA_CHAIN_PROMPT)
        answer = document_chain.invoke({"context": gt_docs,"question":question})
        return recall_ids,answer

    def apply_cal_retriever_score(self,x):
        recall_ids,gt_ids=json.loads(x['retriever_ids']),json.loads(x['relevant_passage_ids'])
        bingo = set(recall_ids) & set(gt_ids)
        recall = len(bingo)/len(gt_ids)
        precision = len(bingo)/len(recall_ids)
        f_score = 2*recall*precision/(recall+precision+1e-30)
        # print(f'{"recall":<10}:{recall:>10}\n{"precision":<10}:{precision:>10}\n{"f_score":<10}:{f_score:>10}')
        return recall,precision,f_score

    def apply_judgy(self,x):
        question = x['question']
        llm_answer = x['llm_answer']
        gt_answer = x['answer']
        score = self.judgy_chain.invoke({"question":question,"gt_answer":gt_answer,"llm_answer":llm_answer})['text']
        score = score.replace('分','')
        try:
            return float(score)
        except:
            return 'nan'

    def ask(self,question,gt_ids=[]):
        retriever = self.db.as_retriever()
        docs = retriever.invoke(question)
        recall_ids = [int(doc.metadata['source']) for doc in docs]

        chain = RetrievalQA.from_llm(self.LLM,retriever=retriever,prompt=self.QA_CHAIN_PROMPT)
        answer = chain.invoke(question)
        
        return recall_ids,answer['result']
    




if __name__ == "__main__":
    knowage = Knowage()
    # answer = knowage.ask('Between which probes does the recurrent translocation breakpoint on chromosome 22 of neuroepithelioma lie?',[2303258])
    # print(answer['answer'])
    # knowage.eval_run()
    knowage.eval_cal()
