# Retrieval-Augmented Generation (RAG) System 

# Import Libraries
# ----------------
import os
from langchain_community.document_loaders import DirectoryLoader,WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
import numpy as np
# from models import OpenaiApi
from data_loader import MyLoader

class Knowage:
    # Constants and API Keys
    # ----------------------
    K = 10
    DOCUMENTS_PATH = "docs"
    VECTOR_DB_DIRECTORY = "VectorStore"
    CHUNK_SIZE = 700
    CHUNK_OVERLAP = 50
    MEMORY_LEN= 5
    QA_CHAIN_PROMPT = PromptTemplate.from_template("""根据下面的上下文（context）内容回答问题。
    如果你不知道答案，就回答不知道，不要试图编造答案。
    答案最多3句话，保持答案简介。
    总是在答案结束时说”谢谢你的提问！“
    {context}
    问题：{question}
    """)

    def version(self):
        return 'base_K10'

    def __init__(self):
        self.embeddings = self.init_embeddings()
        self.db = Chroma(embedding_function=self.embeddings, persist_directory=self.VECTOR_DB_DIRECTORY)
        self.retriever=self.db.as_retriever(search_kwargs={'k':self.K})
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.CHUNK_SIZE, chunk_overlap=self.CHUNK_OVERLAP)
        self.LLM = ChatOllama(model="qwen:14b")
        self.retrieval_qa_chain = self.init_retrieval_qa_chain()

    def init_embeddings(self):
        """Creates embeddings from text."""
        # model_name = "infgrad/stella-base-zh-v3-1792d"
        model_name = "/home/lu/.cache/torch/sentence_transformers/infgrad_stella-base-zh-v3-1792d"

        # model_kwargs = {'device': 'cuda:0'}
        model_kwargs = {'device': 'cuda:0'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                    cache_folder='/home/lu/.cache/torch/sentence_transformers')
        print('init embedding finished')
        return embeddings


    

    def init_retrieval_qa_chain(self):
        """Creates a retrieval QA chain combining model and database."""
        # memory = ConversationBufferWindowMemory(memory_key='chat_history', k=self.MEMORY_LEN, return_messages=True)
        # 要返回source文档，除了加 return_source_documents=True 外，还要在memory处指定 input 和 output 的 key，不然报错
        memory = ConversationBufferWindowMemory(memory_key='chat_history', k=self.MEMORY_LEN, input_key='question', output_key='answer', return_messages=True)
        return ConversationalRetrievalChain.from_llm(self.LLM, retriever=self.retriever, memory=memory,
                                                    combine_docs_chain_kwargs={'prompt': self.QA_CHAIN_PROMPT},
                                                    return_source_documents=True,)

    def add_doc(self,url):
        title = url.split('?')[0].split('-')[-1]
        loader = WebBaseLoader(url)
        pages = loader.load()
        docs = self.text_splitter.split_documents(pages)
        for ii in range(len(docs)):
            docs[ii].metadata['title'] = title
        self.db.add_documents(docs)
    
    def add_doc_local(self,path):
        loader = MyLoader(path)
        self.db.add_documents(loader.docs)

    # def ask(self,question):
    #     answer = self.retrieval_qa_chain.invoke({"question": question})
    #     js_answer = {}
    #     js_answer['question'] = answer['question']
    #     js_answer['answer'] = answer['answer']
    #     js_answer['source_documents'] = [item.json() for item in answer['source_documents']]
    #     return js_answer
    
    def ask(self,question):
        docs = self.retriever.invoke(question)
        print(f'共召回 {len(docs)} 个文档，总计 {np.sum([len(doc.page_content) for doc in docs])} 字。')
        for doc in docs:
            doc.page_content = self.summary_content_with_question(doc,question).content
        document_chain = create_stuff_documents_chain(self.LLM, self.QA_CHAIN_PROMPT)
        answer = document_chain.invoke({"context": docs,"question":question})
        return {'question':question,'answer':answer,'source_documents':docs}
    
    def get_all_title(self):
        metadatas = self.db._collection.get()['metadatas']
        titles = [item['title'] for item in metadatas]
        return list(set(titles))


    def summary_content_with_question(self,content,question):
        prompt = f"""请根据问题，提取下面文本中对回答问题有用的信息。提取信息尽可能简短，高信息密度。
        问题：{question}
        文本：
        {content}
        """
        ret = self.LLM.invoke(prompt)
        return ret


if __name__ == "__main__":
    # main()
    knowage = Knowage()
    # answer = knowage.ask('GPT-API-free 是什么')
    # print(answer['answer'])
    # print(knowage.get_all_title())
    # knowage.add_doc('https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-8-2023')
    # knowage.add_doc('https://github.com/chatanywhere/GPT_API_free')
    
    # answer = knowage.ask('GPT-API-free 是什么')
    # print(answer)
    # print(knowage.get_all_title())

    # knowage.add_doc_local('/home/lu/workspace/public/datawhalechina/')
    # answer = knowage.ask('什么是决策树')
    # print(answer)
    with open('result_tmp.md','w') as f:
        from time import time
        t1 = time()
        question = '决策树和支持向量机有什么不同，如何选择'
        # question = '什么是支持向量机'
        answer = knowage.ask(question)
        f.write(f'# 问题\n{question}\n\n')
        f.write('# 答案\n\n')
        f.write(answer['answer'])
        f.write('\n\n# 参考')
        for ii,doc in enumerate(answer['source_documents'],1):
            f.write(f'\n\n## 参考{ii}\n'+doc.metadata['source']+'\n')
            f.write(doc.page_content)
        print(f'共耗时 {time()-t1} 秒')
