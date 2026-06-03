#!/bin/bash

set -e

echo "========================================"
echo "モデルのダウンロードを開始します: $(date)"
echo "========================================"

echo "[1/3] multilingual-e5-large のダウンロードを開始します..."

hf download intfloat/multilingual-e5-large 
--local-dir /mnt/ssd1tb/k705456/rag/multilingual-e5-large \

> hf_download_multilingual-e5-large.log 2>&1

echo "[1/3] multilingual-e5-large のダウンロードが完了しました: $(date)"

echo "[2/3] ELYZA-japanese-Llama-2-7b-instruct のダウンロードを開始します..."

hf download elyza/ELYZA-japanese-Llama-2-7b-instruct 
--local-dir /mnt/ssd1tb/k705456/rag/ELYZA-japanese-Llama-2-7b-instruct \

> hf_download_ELYZA-japanese-Llama-2-7b-instruct.log 2>&1

echo "[2/3] ELYZA-japanese-Llama-2-7b-instruct のダウンロードが完了しました: $(date)"

echo "[3/3] japanese-large-lm-3.6b-instruction-sft のダウンロードを開始します..."

hf download line-corporation/japanese-large-lm-3.6b-instruction-sft 
--local-dir /mnt/ssd1tb/k705456/rag/japanese-large-lm-3.6b-instruction-sft \

> hf_download_japanese-large-lm-3.6b-instruction-sft.log 2>&1

echo "[3/3] japanese-large-lm-3.6b-instruction-sft のダウンロードが完了しました: $(date)"

echo "========================================"
echo "すべてのモデルのダウンロードが完了しました: $(date)"
echo "========================================"