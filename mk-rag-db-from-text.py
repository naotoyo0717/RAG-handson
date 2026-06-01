# 5.2 FAISSによるデータベースの構築 
# -----①パッセージの作成---------------------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open('joseito.txt', 'r', encoding='utf-8') as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)

texts = text_splitter.split_text(text)

print("【P90作成したチャンク集の確認】")
print(type(texts))
print(len(texts))

print("【P90作成した0番目，１番目のチャンクの確認】")
print(texts[0])
print(texts[1])

# -----②パッセージのベクトル化---------------------------------------------------------
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="/mnt/ssd1tb/k705456/rag/multilingual-e5-large", #各々の環境によって，変更する必要あり．ちなみに．ELYZAは埋め込みモデルではなく，生成モデルなので，ここでは使用できない．
    model_kwargs={'device': 'cuda:0'},
)

d0 = "私は犬が好き。"
d1 = "彼の犬はお利口さん。"
a = embeddings.embed_documents([d0, d1])

print("【P92文書ベクトルの次元数の確認】")
print(type(a[0]))
print(len(a[0]))

# ------③ベクトルデータベースの構築--------------------------------------------------------
from langchain_community.vectorstores import FAISS

db = FAISS.from_texts(texts, embeddings)

db.save_local("joseito.db")

db = FAISS.load_local("joseito.db", embeddings, allow_dangerous_deserialization=True)

a = db.similarity_search("私は犬が好き。")

print("【P95検索された文書の確認】")
print(len(a))
print(type(a[0]))
print(a[0].page_content) #本の通りの結果になるとは限らない．

print("【P95ベクトルによる類似文書の検索例】")
e = embeddings.embed_documents(["私は犬が好き。"])
b = db.similarity_search_by_vector(e[0])
print(b[0].page_content) #本の通りの結果になるとは限らない．