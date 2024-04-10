from sanic import Sanic, response
import requests
from core import Knowage

# 创建 Sanic 应用
app = Sanic("KnowageService")
knowage = Knowage()


titles = []


@app.get("/")
async def main(request):
    return response.html('Hello knowage')

@app.get("/json")
async def js(request):
    return response.json({'result':'hello knowage'})

@app.post("/embedding_doc")
async def embedding_doc(request):
    #### check input
    input = request.json
    print('input###',input)
    urls = input.get('urls')
    if urls is None:
        return response.json({'result': 'param error no urls'})
    if isinstance(urls,str):
        urls = [urls,]
    if not isinstance(urls,list):
        return response.json({'result': 'param error urls only support list or sting'})
    for url in urls:
        if '?' not in url or '-' not in url:
            return response.json({'result': 'param error urls is invalid'})
    #### doc process
    for url in urls:
        knowage.add_doc(url)
        filename = url.split('?')[0].split('-')[-1]
        titles.append(filename)
    return response.json({'result': True})

@app.get("/get_doc")
async def get_doc(request):
    titles = knowage.get_all_title()
    print('servers',titles)
    return response.json({"result":titles})


@app.post("/ask")
async def ask(request):
    input = request.json
    print('question input: ',input)
    question = input.get('question')
    if question:
        answer = knowage.ask(question)
        answer['source_documents'] = [{'page_content':item.page_content,'metadata':item.metadata} for item in answer['source_documents']]
        # print('answer: ',answer)
        return response.json({"result":answer})
    else:
        return response.json({"result":'param question error'})

@app.get("/version")
async def version(request):
    return response.text(knowage.version())




# 启动服务
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8010, workers=1, access_log=False)
