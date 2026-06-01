# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain.chains import RetrievalQA
# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.prompts import PromptTemplate

# model_id =  "/mnt/ssd1tb/k705456/rag/ELYZA-japanese-Llama-2-7b-instruct"



# tokenizer = AutoTokenizer.from_pretrained(
#     model_id,
#     legacy=False,
#     # use_fast=False,
# )

# tokenizer = AutoTokenizer.from_pretrained(
#     model_id,
#     legacy=False,
#     use_fast=False,
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="auto",
#     dtype=torch.float16,
#     low_cpu_mem_usage=True,
# ).eval()

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=300,
#     do_sample=True,
#     temperature=0.1,  
#     repetition_penalty=1.1, 
# )

# qa = RetrievalQA.from_chain_type(
#     llm=HuggingFacePipeline(pipeline=pipe),
#     retriever=retriever,
#     chain_type="stuff",
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": prompt},
#     verbose=True,
# )

# q = "主人公の一番好きな子の名前はなんですか？"
# ans = qa.invoke(q)
# print(ans['result'])


# # template = """
# # ユーザー：以下のテキストを参照して，それに続く質問に答えてください。

# # {context}

# # {question}

# # システム： """

# # prompt = PromptTemplate(
# #     template=template,
# #     input_variables=["context", "question"],
# #     template_format="f-string",
# # )


from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="/mnt/ssd1tb/k705456/rag/multilingual-e5-large", #各々の環境によって，変更する必要あり．ちなみに．ELYZAは埋め込みモデルではなく，生成モデルなので，ここでは使用できない．
    model_kwargs={'device': 'cuda:0'},
    # encode_kwargs={'normalize_embeddings': False}
)

from langchain_community.vectorstores import FAISS

db = FAISS.load_local("joseito.db", embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={"k": 4})

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id =  "/mnt/ssd1tb/k705456/rag/ELYZA-japanese-Llama-2-7b-instruct"
# model_id =  "/mnt/ssd1tb/k705456/rag/japanese-large-lm-3.6b-instruction-sft"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    legacy=False,
    use_fast=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
).eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500,
    temperature=0.1,
    repetition_penalty=1.1,
    # do_sample=True,
)

template = """[INST] <<SYS>>
ユーザー:以下のテキストを参照して，それに続く質問に答えてください．
<</SYS>>
{context}
{question}
[/INST]
システム:"""

# template = """
# ユーザー:以下のテキストを読んで質問に答えてください。

# {context}

# {question}
# システム:"""


from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
    template_format="f-string",
)


from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

qa = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(pipeline=pipe),
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    verbose=True,
)


q = "主人公の一番好きな子の名前はなんですか？"
ans = qa.invoke(q)
print(ans['result'])

print("----------------------------------")

# import re
# pattern = re.compile(r'システム:(.*)', re.DOTALL)
# match = pattern.search(ans['result'])
# ans0 = match.group(1).strip() if match else "回答が見つかりませんでした。"
# print(ans0)

