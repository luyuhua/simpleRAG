import sys
import os
sys.path.insert(0,os.getcwd())
print(sys.path)

from sanic import Sanic
from sanic.response import (json as sanic_json,
                            text as sanic_text,
                            ResponseStream)
from sanic import request
from datetime import datetime
import json
import asyncio
from rag import Rag
from config import DOCS_PATH

app = Sanic("simpleRAG")
# 设置请求体最大为 400MB
app.config.REQUEST_MAX_SIZE = 400 * 1024 * 1024


@app.before_server_start
async def init_local_doc_qa(app, loop):
    rag = Rag()
    app.ctx.rag = rag


async def document(req: request):
    return sanic_text('你好')


async def new_knowledge_base(req: request):
    rag = req.app.ctx.rag
    user_id = req.json.get('user_id')
    kb_name = req.json.get('kb_name')
    collection = rag.db.new_kb(f"{user_id}--{kb_name}",metadata=dict(user_id=user_id,kb_name=kb_name))
    return sanic_json({"code": 200, "msg": f"success create knowledge base {collection.name}",
                       "data": {"kb_id": collection.name, "kb_name": kb_name, "timestamp": datetime.now().strftime("%Y%m%d%H%M")}})


async def upload_files(req: request):
    rag = req.app.ctx.rag
    files = req.files.getlist('files')
    kb_id = req.form.get('kb_id')
    user_id = req.form.get('user_id')
    root_path = os.path.join(DOCS_PATH,user_id)
    os.makedirs(root_path,exist_ok=True)
    filelist = []
    for file in files:
        file_path = os.path.join(root_path,file.name)
        with open(file_path,'wb') as f:
            f.write(file.body)
        filelist.append(os.path.abspath(file_path))
    data = rag.db.add_docs(docs=filelist,kb_name=kb_id,user_id=user_id)
    return sanic_json({"code": 200, "msg": "success，后台正在飞速上传文件，请耐心等待","data":  data})


async def local_doc_chat(req: request):
    rag = req.app.ctx.rag
    # user_id = req.json.get('user_id')
    history = req.json.get('history')
    kb_names = req.json.get('kb_ids')
    streaming = req.json.get('streaming')
    question = req.json.get('question')
    answer = rag.ask(question=question,kb_names=kb_names,history=history,stream=streaming)
    source_documents = [dict(file_id=doc.id_,
                            file_name=doc.metadata['file_name'],
                            content=doc.text,
                            retrieval_query=question,
                            score=doc.score,
                            embed_version='bge1.5-zh') 
                            for doc in answer.source_nodes]
    if not streaming:
        history = history.append([question,answer.response])
        data = dict(code=200,msg='success',question=question,response=answer.response,history=history,source_documents=source_documents)
        return sanic_json(data)
    else:
        async def generate_answer(response):
                results = ''
                for result in answer.response_gen:
                    results += result
                    data = {'code': 200, 'msg': 'success', 'question': '', 'response': result, 'history': [], 'source_documents': source_documents}
                    await response.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n")
                    await asyncio.sleep(0.001) # 类似清缓存操作，https://community.sanicframework.org/t/confusion-with-stream-response-of-sanic/1235
                history.append([question,results])
                data = {'code': 200, 'msg': 'success', 'question': '', 'response': '', 'history': history, 'source_documents': source_documents}
                await response.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n")
                await response.eof()
        response_stream = ResponseStream(generate_answer, content_type='text/event-stream')
        return response_stream        


async def list_kbs(req: request):
    rag = req.app.ctx.rag
    user_id = req.json.get('user_id')
    collections = rag.db.list_kb()
    data = [{"kb_id": collection.name, "kb_name": collection.metadata["kb_name"]} for collection in collections if collection.metadata["user_id"]==user_id]
    return sanic_json({"code": 200, "data": data})


async def list_docs(req: request):
    rag = req.app.ctx.rag
    kb_id = req.json.get('kb_id')
    data = rag.db.list_doc(kb_name=kb_id)
    status_count = { 
        "green": len([item for item in data if item['status']=='green']),
        "red": len([item for item in data if item['status']=='red']),
        "gray": len([item for item in data if item['status']=='gray']),
        "yellow": len([item for item in data if item['status']=='yellow']),
		}
    return sanic_json({"code": 200, "msg": "success", "data": {'total': status_count, 'details': data}})


async def delete_docs(req: request):
    rag = req.app.ctx.rag
    kb_id = req.json.get('kb_id')
    doc_ids = req.json.get('file_ids')
    rag.db.del_doc(kb_name=kb_id,doc_ids=doc_ids)
    return sanic_json({"code": 200, "msg": f"documents {doc_ids} delete success"})


async def delete_knowledge_base(req: request):
    rag = req.app.ctx.rag
    kb_ids= req.json.get('kb_ids')
    rag.db.del_kb(kb_names=kb_ids)
    return sanic_json({"code": 200, "msg": f"Knowledge Base {kb_ids} delete success"})


app.add_route(document, "/api/docs", methods=['GET'])
app.add_route(new_knowledge_base, "/api/local_doc_qa/new_knowledge_base", methods=['POST'])  # tags=["新建知识库"]
# app.add_route(upload_weblink, "/api/local_doc_qa/upload_weblink", methods=['POST'])  # tags=["上传网页链接"]
app.add_route(upload_files, "/api/local_doc_qa/upload_files", methods=['POST'])  # tags=["上传文件"] 
app.add_route(local_doc_chat, "/api/local_doc_qa/local_doc_chat", methods=['POST'])  # tags=["问答接口"] 
app.add_route(list_kbs, "/api/local_doc_qa/list_knowledge_base", methods=['POST'])  # tags=["知识库列表"] 
app.add_route(list_docs, "/api/local_doc_qa/list_files", methods=['POST'])  # tags=["文件列表"]
# app.add_route(get_total_status, "/api/local_doc_qa/get_total_status", methods=['POST'])  # tags=["获取所有知识库状态"]
# app.add_route(clean_files_by_status, "/api/local_doc_qa/clean_files_by_status", methods=['POST'])  # tags=["清理数据库"]
app.add_route(delete_docs, "/api/local_doc_qa/delete_files", methods=['POST'])  # tags=["删除文件"] 
app.add_route(delete_knowledge_base, "/api/local_doc_qa/delete_knowledge_base", methods=['POST'])  # tags=["删除知识库"] 
# app.add_route(rename_knowledge_base, "/api/local_doc_qa/rename_knowledge_base", methods=['POST'])  # tags=["重命名知识库"] 


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8777, workers=2, access_log=False)
    app.run(host='0.0.0.0', port=8777, workers=1, access_log=True)
