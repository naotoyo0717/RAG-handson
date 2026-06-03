from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import torch

embeddings = HuggingFaceEmbeddings(
    model_name="/mnt/ssd1tb/k705456/rag/multilingual-e5-large", #各々の環境によって，変更する必要あり．ちなみに．ELYZAは埋め込みモデルではなく，生成モデルなので，ここでは使用できない．
    model_kwargs={'device': 'cuda:0'},
    # encode_kwargs={'normalize_embeddings': False}
)

db = FAISS.load_local("yamatano.db", embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={"k": 4})


# -----②ローカルLLMを使ったpipelineインスタンスの設定---------------------------------------------------------

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
    return_full_text=False,  #余計な文章を生成しないようにするオプション
)

# -----①RAG用のテンプレート---------------------------------------------------------

# このプロンプトテンプレートはELYZA専用
template = """[INST] <<SYS>>
あなたは質問応答システムです。

与えられた文書のみを根拠として回答してください。
回答だけを出力してください。
<</SYS>>

参考文書:
{context}

質問:
{question}

[/INST]
"""

# このプロンプトテンプレートはjapanese-large-lm-3.6b-instruction-sft専用
# template = """
# ユーザー:以下のテキストを読んで質問に答えてください。

# {context}

# {question}
# システム:"""

# -----①PromptTemplateによるプロンプト生成器の設定---------------------------------------------------------

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
    template_format="f-string",
)

# -----②ローカルLLMを使ったRetrievalQAの設定---------------------------------------------------------

qa = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(pipeline=pipe),
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    verbose=True,
)

# -----②ローカルLLMを使ったRAGの実行例---------------------------------------------------------

q = "主人公が勤務している食品メーカーの名前は?"
ans = qa.invoke(q)

print("===== 参照したテキスト =====")

for i, doc in enumerate(ans["source_documents"], start=1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)

print("\n===== 回答 =====")
print(ans["result"])

