import requests
import json
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatZhipuAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from zhipuai import ZhipuAI

client = ZhipuAI(api_key="4a88699c5d2478ee7e5b8cf7b46d12be.XOC8sBv8ceAcws1N") # 填写您自己的APIKey

def get_last_version_answer():
    questions = ["什么是决策树",
                "决策树和支持向量机有什么不同，如何选择"]

    url = 'http://127.0.0.1:8010/'
    version = requests.get(url = url+'version').text

    answers = []
    for question in questions:
        answer = requests.post(url = url+'ask',json= {'question': question}).json()
        answers.append(answer['result'])
    with open('experiment/results/'+version+'.json','w') as f:
        json.dump(answers,f,ensure_ascii=False)
    print('end')

def judgy():
    with open('experiment/results/base.json','r') as f:
        js1 = json.load(f)
    with open('experiment/results/base_K20.json','r') as f:
        js2 = json.load(f)
    for item1,item2 in zip(js1,js2):
        question = item1['question']
        content1 = '\n'.join([item['page_content'] for item in item1['source_documents']])
        content2 = '\n'.join([item['page_content'] for item in item2['source_documents']])
        prompt = f"""
【背景】我们正在评估搜索引擎的召回能力。
【任务】我将给你一个query，以及针对这个query，两个不同的搜索引擎召回的内容。你将评估比较两个搜索引擎召回的内容的优劣情况。
【要求】
评估需要考虑一下几个方面的因素
1-召回的内容是否有尽可能多的对回答query有用的信息。有用信息越多，效果越好。
2-召回的内容是否有很多不相关的信息。不相关的信息越多，效果越差。
3-召回的信息是否足球精简，同一信息约精简，效果越好，反之，越冗余效果越差。
你回答的内容需要逐步分析上面的每一点要求，然后再给出哪个召回内容更好的结论。
【query】
###
{question}
###
【召回内容1】
###
{content1}
###
【召回内容2】
###
{content2}                      
###
"""
        response = client.chat.completions.create(
            model="glm-4",  # 填写需要调用的模型名称
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        ret = response.choices[0].message
        print(ret)



if __name__ == '__main__':
    # get_last_version_answer()
    ret = judgy()
    print(ret)

