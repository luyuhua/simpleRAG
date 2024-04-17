import os 
import sqlite3
import chromadb
from pathlib import Path
import uuid
from datetime import datetime
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import SQLDB_PATH,VECTORDB_PATH
from loader import Loader


class DB:
    def __init__(self) -> None:
        os.makedirs(os.path.dirname(SQLDB_PATH),exist_ok=True)
        self.conn = sqlite3.connect(SQLDB_PATH)
        self.cursor = self.conn.cursor()
        self.create_sql_table()
        self.chroma_client = chromadb.PersistentClient(path= VECTORDB_PATH)
        self.loader = Loader()
    
    def create_sql_table(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS docs
                            (doc_id          CHAR(50) PRIMARY KEY     NOT NULL,
                            doc_name         CHAR(100)    NOT NULL,
                            user_id          CHAR(50)     NOT NULL,
                            kb_name          CHAR(50)     NOT NULL,
                            bytes            int          NOT NULL,
                            status           CHAR(8)      NOT NULL,   
                            length           int          NOT NULL,
                            timestamp        CHAR(50)     NOT NULL
                            );''')
        print ("数据表创建成功")
        self.conn.commit()

    ## 创建知识库
    def new_kb(self,kb_name,metadata=None):
        collection = self.chroma_client.create_collection(name=kb_name,metadata=metadata)
        return collection
    
    ## 查看知识库
    def list_kb(self):
        collections = self.chroma_client.list_collections()
        return collections

    ## 删除知识库
    def del_kb(self,kb_names):
        if isinstance(kb_names,list):
            kb_names=[kb_names,]
        for kb_name in kb_names:
            self.chroma_client.delete_collection(name=kb_name)
            self.del_docs(filter={'kb_name':kb_name})

    ## 添加文档
    def add_docs(self,docs,kb_name,user_id):
        is_ins = [self.select(filter=dict(doc_name=os.path.basename(doc),kb_name=kb_name,user_id=user_id)) for doc in docs]
        docs = [doc for doc,is_in in zip(docs,is_ins) if len(is_in)==0]
        doc_ids = self.insert(docs,kb_name,user_id)
        # state = self.doc_state(doc_ids)
        documents = self.loader.load_data(docs=docs)

        splitter = SentenceSplitter()
        nodes = splitter.get_nodes_from_documents(documents)
        index = self.get_index(kb_name=kb_name)
        index.insert_nodes(nodes=nodes)

        self.updata(kv={"status":"green"},filter={"doc_id":doc_ids})
        state = self.doc_state(doc_ids)
        return state

    def get_index(self,kb_name):
        collection = self.chroma_client.get_collection(name=kb_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        return index

    ## 查看文档
    def list_doc(self,kb_name):
        doc_ids = self.get_docs(filter={'kb_name':kb_name})
        data = self.doc_state(doc_ids=doc_ids)
        return data

    ## 删除文档
    def del_doc(self,kb_name,doc_ids):
        if not isinstance(doc_ids,list):
            doc_ids=[doc_ids,]
        collection = self.chroma_client.get_collection(name=kb_name)
        collection.delete(ids=doc_ids)
        self.del_docs(filter={"doc_id":doc_ids})
    

    ###### sqldb相关操作 ######
    def insert(self,docs,kb_name,user_id):
        ret = []
        for doc in docs:
            doc = doc if isinstance(doc,Path) else Path(doc)
            doc_id = 'DOC'+uuid.uuid4().hex
            sql = f"""INSERT INTO `docs` 
            (`doc_id`, `doc_name`, `user_id`, `kb_name`, `bytes`, `status`, `length`,`timestamp`) 
            VALUES 
            ("{doc_id}","{doc.name}", "{user_id}", "{kb_name}", {os.path.getsize(doc)}, "gray", 0,"{datetime.now().strftime("%Y%m%d%H%M")}") 
            """
            self.cursor.execute(sql)
            ret.append(doc_id)
        self.conn.commit()
        return ret

    def doc_state(self,doc_ids):
        if len(doc_ids) == 0:
            return []
        doc_ids_str = str(doc_ids).replace('[','(').replace(']',')')
        sql = f"""select doc_id,doc_name,status,bytes,timestamp,length from docs where doc_id in {doc_ids_str}"""
        self.cursor.execute(sql)
        sql_ret = self.cursor.fetchall()
        ret = [dict(file_id=item[0],
                    file_name=item[1],
                    status = item[2],
                    bytes = item[3],
                    timestamp = item[4],
                    content_length = item[5]
                    ) for item in sql_ret]
        return ret
    
    def get_docs(self,filter={}):
        if len(filter)==0:
            sql = """select doc_id from docs"""
        else:
            sql = """select doc_id from docs where \n"""
            sql += self._get_condition_sql(filter=filter)
        self.cursor.execute(sql)
        sql_ret = self.cursor.fetchall()
        return [item[0] for item in sql_ret]
    
    def del_docs(self,filter):
        sql = """delete from docs where \n"""
        sql += self._get_condition_sql(filter=filter)
        self.cursor.execute(sql)
        self.conn.commit()
    
    def updata(self,kv,filter):
        sql = """update docs set \n """
        for k,v in kv.items():
            if isinstance(v,str):
                sql += f" {k} = '{v}',"
            else:
                sql += f" {k} = {v},"
        sql = sql[:-1] + ' where \n '
        sql += self._get_condition_sql(filter=filter)
        self.cursor.execute(sql)
        self.conn.commit()

    def select(self,row_str='*',filter={}):
        sql = f"""select {row_str} from docs \n """
        sql = sql[:-1] + ' where \n '
        sql += self._get_condition_sql(filter=filter)
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def _get_condition_sql(self,filter):
        sql = ''
        for k,v in filter.items():
            if isinstance(v,str):
                sql += f"{k}='{v}' \n and "
            elif isinstance(v,list):
                sql += f"{k} in {str(v).replace('[','(').replace(']',')')} \n and "
            else:
                sql += f"{k}={v} \n and "
        sql = sql[:-4]      
        return sql  
