import requests
from bs4 import BeautifulSoup

url = 'https://www.aozora.gr.jp/cards/000035/files/275_13903.html'

response = requests.get(url)

soup = BeautifulSoup(response.content, 'html.parser')
# パーサでhtmlファイルを解析

for script_or_style in soup(['script', 'style']):
    script_or_style.extract()
# scriptやstyleタグを削除

text = soup.get_text()
# テキスト部分を抽出

lines = (line.strip() for line in text.splitlines())
# 改行で区切って、先頭と末尾の空白を削除

chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

text = '\n'.join(chunk for chunk in chunks if chunk)

outtext = 'joseito.txt'
with open(outtext, 'w', encoding='utf-8') as file:
    file.write(text)

print(url, "の内容を", outtext, "に出力しました。")