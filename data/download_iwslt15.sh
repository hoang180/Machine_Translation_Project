#!/usr/bin/env bash
# Script: data/download_iwslt15.sh
# Phụ trách: Phần 1 (Dữ liệu)
# Tự động tải tập dữ liệu IWSLT'15 En-Vi từ nguồn Stanford NLP

set -e

DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

BASE_URL="https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi"

FILES=(
    "train.en"
    "train.vi"
    "tst2012.en"
    "tst2012.vi"
    "tst2013.en"
    "tst2013.vi"
    "vocab.en"
    "vocab.vi"
)

echo "=========================================================="
echo "BẮT ĐẦU TẢI DỮ LIỆU IWSLT'15 EN-VI (STANFORD NLP)"
echo "Thư mục đích: $DATA_DIR"
echo "=========================================================="

for FILE in "${FILES[@]}"; do
    TARGET="$DATA_DIR/$FILE"
    if [ -f "$TARGET" ]; then
        echo "[EXISTS] $FILE đã tồn tại, bỏ qua."
    else
        echo "[DOWNLOADING] $FILE từ $BASE_URL/$FILE ..."
        curl -fSL "$BASE_URL/$FILE" -o "$TARGET"
    fi
done

echo "=========================================================="
echo "TẢI DỮ LIỆU HOÀN TẤT!"
echo "=========================================================="
