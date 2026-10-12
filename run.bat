@echo off
echo Starting ETF Basket vs. Box Arbitrage Scanner on http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
