from langchain_core.documents import Document

path = '/home/lu/workspace/public/datawhalechina/notebook/C1 大模型简介/1. 什么是⼤模型.md'
with open(path,'r') as f:
    lines = f.readlines()

def proces_level(level_info,level):
    if level == len(level_info):
        level_info[-1] += 1
    elif level == len(level_info) + 1:
        level_info.append(1)
    elif level < len(level_info):
        level_info = level_info[:level]
        level_info[-1] += 1
    else:
        raise 'level info error'
    return level_info

docs = []
text = ''
title = ''
level_info = []
lines.append('#') # 触发最后chapter
for line in lines:
    if line and line[0]=='#':
        if level_info:
            doc = Document(text)
            doc.metadata={'source':'xxx','title':title,'chapter':level_info.copy()}
            docs.append(doc)


        level = line.split(' ')[0].count('#')
        level_info = proces_level(level_info,level)
        title = ''.join(line.split(' ')[1:])
        text = line 
    else:
        text = text + '\n' + line


print('end')

from langchain_core import Document 
a = Document()