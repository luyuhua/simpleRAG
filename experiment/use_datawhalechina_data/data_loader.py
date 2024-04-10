from __future__ import annotations
import fitz 
from langchain_core.documents import Document
from glob import glob
import os


class MyLoader:
    FILE_TYPES = ['pdf','md']

    docs:list = []

    def __init__(self,path) -> None:
        if os.path.isdir(path):
            self.process_directory(path)
        elif os.path.isfile(path):
            self.process_file(path)
        else:
            raise 'path error'
        print('end')


    def process_file(self,path):
        file_type = path.split('.')[-1]
        if file_type == 'pdf':
            self.docs.extend(self.parse_pdf(path))
        elif file_type == 'md':
            self.docs.extend(self.parse_md(path))
        else:
            print(f'sorry {file_type} type file is not support now')
    
    def process_directory(self,directory,recursive=True):
        filelist = []
        for file_type in self.FILE_TYPES:
            filelist.extend(glob(os.path.join(directory,'**',f'*.{file_type}'),recursive=recursive))
        for file in filelist:
            self.process_file(file)
    
    def process_chapter_info(self,chapter_info,level):
        if level == len(chapter_info):
            chapter_info[-1] += 1
        elif level < len(chapter_info):
            chapter_info = chapter_info[:level]
            chapter_info[-1] += 1
        else:
            for _ in range(level-len(chapter_info)):
                chapter_info.append(1)
        return chapter_info
    
    def parse_pdf(self,path) -> dict:
        print(os.path.basename(path))
        pdf = fitz.open(path)
        topics = pdf.get_toc(simple=False)
        # 创建最后一个虚拟的topic，使得最后的chapter可以读完全文
        topics.append([None,None,pdf.page_count,{'to':fitz.Point(0,0)}])

        docs = []
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
            
            chapter_info = self.process_chapter_info(chapter_info,level)
            doc = Document(text)
            doc.metadata = {'source':path,'title':title1,'chapter':chapter_info.copy()}
            docs.append(doc)
        return docs
    
    def parse_md(self,path):
        print(os.path.basename(path))
        with open(path,'r') as f:
            lines = f.readlines()
        docs = []
        text = ''
        title = ''
        chapter_info = []
        lines.append('#') # 触发最后chapter
        for line in lines:
            if line and line[0]=='#':
                if chapter_info:
                    doc = Document(text)
                    doc.metadata={'source':path,'title':title,'chapter':chapter_info.copy()}
                    docs.append(doc)
                level = line.split(' ')[0].count('#')
                chapter_info = self.process_chapter_info(chapter_info,level)
                title = ''.join(line.split(' ')[1:])
                text = line 
            else:
                text = text + '\n' + line   
        return docs


    @staticmethod
    def new_chapter(title) -> dict:
        return {'title':title,'childs':[],'text':None}

    @classmethod
    def parse_pdf_to_dict(cls,path) -> dict:
        pdf = fitz.open(path)
        file_name = os.path.basename(pdf.name)
        topics = pdf.get_toc(simple=False)
        # 创建最后一个虚拟的topic，使得最后的chapter可以读完全文
        topics.append([None,None,pdf.page_count,{'to':fitz.Point(0,0)}])

        print(file_name,topics[0])
        meta_dic = cls.new_chapter(file_name)
        
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

            cur = meta_dic
            for _ in range(level-1):
                cur = cur['childs'][-1]
            cur['childs'].append(cls.new_chapter(title1))
            cur['childs'][-1]['text'] = text
        return meta_dic


    



if __name__ == "__main__":
    # path = '/home/lu/workspace/public/datawhalechina/pumpkin_book.pdf'
    path = '/home/lu/workspace/public/datawhalechina/'
    loader = MyLoader(path)

    print('end')






