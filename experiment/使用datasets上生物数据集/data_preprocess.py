import pandas as pd
import json

# 取 topK 行content
def mini_passage(topK=2000):
    df = pd.read_parquet('/home/lu/.cache/huggingface/datasets/mini-bioasq/data/passages.parquet/part.0.parquet')
    print(f'原始passages,共 {len(df)} 行')
    df.drop(df[df['passage']=='nan'].index,inplace=True)
    print(f'去除nan后，共 {len(df)} 行')
    ind_topK = df.index.to_list()[topK]
    mini_df = df[df.index<ind_topK]
    mini_df.to_csv('experiment/mini_passages.csv')
    print(f'topK={topK},id={ind_topK}')
    return ind_topK

def mini_qa():
    mini_passage_df = pd.read_csv('experiment/mini_passages.csv',index_col='id')
    max_id = mini_passage_df.index.to_list()[-1]
    df = pd.read_parquet('/home/lu/.cache/huggingface/datasets/mini-bioasq/data/test.parquet/part.0.parquet')
    print(f'原始qa,共 {len(df)} 个')

    def map_fun(x):
        ids = json.loads(x)
        for id in ids:
            if id>max_id:
                return 'nan'
            if id not in mini_passage_df.index:
                return 'nan'
        return x
    
    df['relevant_passage_ids'] = df['relevant_passage_ids'].map(map_fun)
    mini_df = df.drop(df[df['relevant_passage_ids']=='nan'].index)
    mini_df.to_csv('experiment/mini_qas.csv')
    print(f'mini qa,共 {len(mini_df)} 个')
    return 




if __name__ == "__main__":
    mini_passage()
    mini_qa()
