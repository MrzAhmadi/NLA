#!/bin/bash
set -e

echo "=== Step 1: Generate activation dataset ==="
python pipeline/generate_data.py

echo ""
echo "=== Step 2: Generate explanations (local model) ==="
python pipeline/generate_explanations.py

echo ""
echo "=== Step 3: Train AV + AR ==="
python pipeline/train.py

echo ""
echo "=== Step 4: Evaluate ==="
python pipeline/evaluate.py

echo ""
echo "=== Step 5: Launch Streamlit app ==="
python -m streamlit run app.py

