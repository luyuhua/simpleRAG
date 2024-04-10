from __future__ import annotations
import fitz 
from langchain_core.documents import Document
from glob import glob
import os
import numpy as np
import json
from collections import OrderedDict,defaultdict
import copy


class MyLoader:
    FILE_TYPES = ['pdf','md']
    docs:list = []

    def __init__(self,path,chunk_size=1000,over_size=100,recursive=True) -> None:
        if chunk_size<=over_size: raise Exception('chunk_size or over size error')
        self.chunk_size = chunk_size
        self.over_size = over_size
        if os.path.isfile(path):
            filelist = [path,]
        elif os.path.isdir(path):
            filelist = []
            for file_type in self.FILE_TYPES:
                filelist.extend(glob(os.path.join(path,'**',f'*.{file_type}'),recursive=recursive))
        else:
            raise 'path error'
        for file in filelist:
            self.docs.extend(self.parse(file))
    

    def parse(self,path):
        file_type = path.split('.')[-1]
        if file_type == 'pdf':
            docs =  self.parse_pdf(path)
        elif file_type == 'md':
            docs = self.parse_md(path)
        else:
            raise f'sorry {file_type} type file is not support now'  
        for k in docs.keys():
            docs[k]['content'] = docs[k]['text'] 
            docs[k]['merge_chapters'] = []
            docs[k]['is_used'] = 0

        docs = self.merge(docs) 
        docs = self.split(docs)    
        docs = self.docs2Documents(docs)  
        return docs      

    
    def parse_pdf(self,path) -> dict:
        print(os.path.basename(path))
        pdf = fitz.open(path)
        topics = pdf.get_toc(simple=False)
        # 创建最后一个虚拟的topic，使得最后的chapter可以读完全文
        topics.append([None,None,pdf.page_count,{'to':fitz.Point(0,0)}])
        docs = OrderedDict()
        docs[str([])]={'text':'','source':path,'title':''}
        chapter_info = []
        for toc1,toc2 in zip(topics[:-1],topics[1:]):
            level,title1,page_num1,meta1 = toc1
            _,title2,page_num2,meta2 = toc2
            page_num1 -= 1
            page_num2 -= 1
            page1 = pdf[page_num1]
            page2 = pdf[page_num2]

            if 'to' in meta1:
                y0 = page1.rect.y1-meta1['to'].y
                y1 = page2.rect.y1-meta2['to'].y
            else: 
                # 当一处页面有多个和标题一致的文字时，默认最大返回覆盖内容（后期可根据font智能选择）
                y0 = page1.search_for(title1[:10])[0].y0 # 如果title太长，在pdf中显示两行的话，search就无法搜索到
                y1 = page2.search_for(title2[:10])[-1].y0 if title2 else page2.rect.y1 # 最后一页特殊处理

            if page_num1 == page_num2:
                rect = fitz.Rect(0,y0,page1.rect.x1,y1)
                text = page1.get_textbox(rect)
            else:
                rect = fitz.Rect(0,y0,page1.rect.x1,page1.rect.y1)
                text = page1.get_textbox(rect)
                for page_num in range(page_num1+1,page_num2-1):
                    text += pdf[page_num].get_text()
                rect = fitz.Rect(0,0,page2.rect.x1,y1)
                text += page2.get_textbox(rect)
            
            chapter_info,parent_info = self.next_chapter_info(chapter_info,level)
            for item in parent_info:# 加入空的父节点(正常文章无空父节点)
                docs[str(item)]={'text':'','source':path,'title':''}
            doc = {'text':text,'source':path,'title':title1}
            docs[str(chapter_info)] = doc    
        return docs
    

    def parse_md(self,path):
        print(os.path.basename(path))
        with open(path,'r') as f:
            lines = f.readlines()
        # docs = []
        docs = OrderedDict()
        docs[str([])]={'text':'','source':path,'title':''}
        text = ''
        title = ''
        chapter_info = []
        lines.append('#') # 触发最后chapter
        if lines[0][0] != '#': # 防止有些md不按套路出牌，全程无 #
            lines.insert(0,'# \n')
        for line in lines:
            if line and line[0]=='#':
                if chapter_info:
                    doc = {'text':text,'source':path,'title':title}
                    docs[str(chapter_info)] = doc
                level = line.split(' ')[0].count('#')
                chapter_info,parent_info = self.next_chapter_info(chapter_info,level)
                for item in parent_info:# 加入空的父节点(正常文章无空父节点)
                    docs[str(item)]={'text':'','source':path,'title':''}
                title = ''.join(line.split(' ')[1:])
                text = line 
            else:
                text = text + '\n' + line     
        return docs


    def next_chapter_info(self,chapter_info,level):
        parent_info = []
        if level == len(chapter_info):
            chapter_info[-1] += 1
        elif level < len(chapter_info):
            chapter_info = chapter_info[:level]
            chapter_info[-1] += 1
        elif level == len(chapter_info) +1:
            chapter_info.append(1)
        else:
            for _ in range(level-len(chapter_info)):
                chapter_info.append(1)
                parent_info.append(chapter_info.copy())
            parent_info.pop()
        return chapter_info,parent_info
    

    def merge(self,docs):
        kv_list = list(docs.items())
        ii=0
        while ii<len(kv_list)-1:
            current_content = kv_list[ii][1]['content']
            merge_chapters = [json.loads(kv_list[ii][0]),]
            # 防止数组越界，最后一个等等单独处理
            # 父子节点，兄弟节点可合并。其他情况不能合并
            # 判断是合并后长度是否满足要求
            while ii < len(kv_list)-1 \
                  and len(json.loads(kv_list[ii][0])) <= len(json.loads(kv_list[ii+1][0]))  \
                  and len(current_content+kv_list[ii+1][1]['content']) < self.chunk_size:
                current_content += kv_list[ii+1][1]['content']
                merge_chapters.append(json.loads(kv_list[ii+1][0]))
                ii+=1
            kv_list[ii][1]['content'] = current_content
            kv_list[ii][1]['merge_chapters'] = merge_chapters
            kv_list[ii][1]['is_used'] = 1 if len(current_content)>0 else 0 #防止空的根节点加入
            ii+=1
        kv_list[len(kv_list)-1][1]['is_used'] = 1
        return docs

    
    def split(self,docs):
        new_docs = OrderedDict()
        for k,v in docs.items():
            if v['is_used']==1 and len(v['content'])>self.chunk_size:
                for ii in range(len(v['content'])//(self.chunk_size-self.over_size) +1 ):
                    new_k = str(json.loads(k)+[-1*(ii+1),]) #用 -n 区别于正常的子目录
                    new_v = copy.deepcopy(v)
                    new_v['content'] = v['content'][ii*(self.chunk_size-self.over_size):(ii+1)*(self.chunk_size-self.over_size)+self.over_size]
                    new_v['title'] = v['title'] + f' -- {ii+1}'
                    new_docs[new_k] = new_v
            else:
                new_docs[k] = v
        return new_docs


    def docs2Documents(self,docs):
        documents = []
        for k,v in docs.items():
            if v['is_used']==1 and len(v['content'])>0:
                doc = Document(v['content'])
                if len(v['merge_chapters'])>1:
                    title = str([docs[str(chapter)]['title'] for chapter in v['merge_chapters']])
                    chapter = str(v['merge_chapters'])
                    doc.metadata = {'source':v['source'],'title':title,'chapter':chapter}
                else:
                    doc.metadata = {'source':v['source'],'title':v['title'],'chapter':k}
                documents.append(doc)
        return documents


if __name__ == "__main__":
    # path = '/home/lu/workspace/public/datawhalechina/pumpkin_book.pdf'
    path = '/home/lu/workspace/public/datawhalechina/'
    loader = MyLoader(path,chunk_size=100,over_size=1000)
    print('end')






