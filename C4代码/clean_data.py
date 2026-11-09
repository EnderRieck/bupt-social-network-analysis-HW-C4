import pandas as pd
import random
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import csv

data_path = "/your_path/cleaned_shoe_data.csv"
df = pd.read_csv(data_path)#读取女鞋数据集

documents = df.apply(lambda row: 
    f"品牌: {row['brand']}，"
    f"名称: {row['name']}，"
    f"历史最高价: {row.get('prices.amountMax', '')}，"
    f"历史最低价: {row.get('prices.amountMin', '')}，"
    f"颜色: {row.get('prices.color', '')}，"
    f"类别: {row.get('categories', '')}，"
    f"尺码: {row.get('sizes', '')}，"
    f"类别: {row.get('prices_merchant', '')}，"
    , axis=1).tolist() #将女鞋数据集按属性合并

# 载入Embedding模型
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 创建或加载Chroma向量数据库
persist_directory = "./chroma_shoe_db"
vectordb = Chroma.from_texts(
    texts=documents,
    embedding=embedding_model,
    persist_directory=persist_directory
)

input_file_data="/data1/wuquanda/agent_start/data/cleaned_shoe_data.csv"

df = pd.read_csv(input_file_data)
texts = []
metadatas = []
i=2
for _,row in df.iterrows():
    low_price=row.get('prices.amountMin', 0)
    high_price=row.get('prices.amountMax', 0)
    time=row.get('dateUpdated', '未知')
    dt = datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")
    year_month = dt.strftime("%Y-%m")
    discount='是' if random.random()<0.4 else '否'
    discount_type='null'
    if discount=='是':
        discount_=round((random.random()+1)/2,1)
        discount_type='满减' if random.random()<0.5 else discount_
    merchants=['淘宝','京东','拼多多']
    if low_price!=high_price:
        price= random.randint(round(low_price),round( high_price))
        if discount_type=='满减':
            d_price=random.randint(round(0.1*price),round(0.2*price))
            s_price=round(0.9*price)
            discount_type=f'满{s_price}减{d_price}'
        texts.append(
                f"品牌: {row['brand']},名称: {row['name']},颜色: {row.get('prices.color', '')},价格: {price}；"
                f"是否折扣: {discount}；折扣类型:{discount_type}；在售商店: {merchants[i]}；"
                f"时间: {year_month}；类别: {row.get('categories', '')}；"
            )
            
            # metadata 里存 list
        metadatas.append({
                "brand": str.lower(row["brand"]),
                "name": str.lower(row["name"]),
                "color": str.lower(row.get('prices.color', '')),
                'price': price,
                'categories': "Women's shoe",
                'time': year_month,
                'discount': discount,
                'discount_type': discount_type,
                'prices_merchant': merchants[i],
                # 其他字段...
            })
    else:
        price=low_price
        if discount_type=='满减':
            d_price=random.randint(round(0.1*price),round(0.2*price))
            s_price=round(0.9*price)
            discount_type=f'满{s_price}减{d_price}'
        texts.append(
                f"品牌: {row['brand']},名称: {row['name']},颜色: {row.get('prices.color', '')},价格: {price}；"
                f"是否折扣: {discount}；折扣类型:{discount_type}；在售商店: {merchants[i]}；"
                f"时间: {year_month}；类别: {row.get('categories', '')}；"
            )
            
            # metadata 里存 list
        metadatas.append({
                "brand": str.lower(row["brand"]),
                "name": str.lower(row["name"]),
                "color": str.lower(row.get('prices.color', '')),
                'price': low_price,
                'categories': "Women's shoe",
                'time': year_month,
                'discount': discount,
                'discount_type': discount_type,
                'prices_merchant': merchants[i],
                # 其他字段...
            })
# 载入Embedding模型
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 创建或加载Chroma向量库
persist_directory = "./pdd_shoe_chroma_dbs"
vectordb = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    metadatas=metadatas,
    persist_directory=persist_directory
)

# 持久化数据库
vectordb.persist()
result_csv_path='./pdd_shoe_chroma_dbs/pdd_metadata.csv'
results_df = pd.DataFrame(metadatas)
results_df.to_csv(
    result_csv_path,
    index=False,
    sep=",",                       # 默认分隔符
    quotechar='"',                 # 用双引号包裹
    quoting=csv.QUOTE_MINIMAL,     # 遇到逗号、换行才加引号
    encoding="utf-8"
)
#results_df.to_csv(result_csv_path, index=False)
