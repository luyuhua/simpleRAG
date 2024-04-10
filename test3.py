# import requests
# import json

# url = "http://127.0.0.1:8777/api/local_doc_qa/list_knowledge_base"
# headers = {
#     "Content-Type": "application/json"
# }
# data = {
#     "user_id": "zzp"
# }

# response = requests.post(url, headers=headers, data=json.dumps(data))

# print(response.status_code)
# print(response.json())


import mysql.connector
from mysql.connector import pooling

host = '127.0.0.1'
port = 3306
user = 'root'
password = '123456'
database = 'qanything'


def check_database_(host, port, user, password, database_name):
    # 连接 MySQL 服务器
    cnx = mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password
    )
    
    # 检查数据库是否存在
    cursor = cnx.cursor(buffered=True)
    cursor.execute('SHOW DATABASES')
    databases = [database[0] for database in cursor]


check_database_(host, port, user, password, database)
