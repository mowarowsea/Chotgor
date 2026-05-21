FROM python:3.11-slim

WORKDIR /app

# システム依存パッケージ（pyarrow / lancedb 等の C ライブラリ用）
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python依存パッケージ
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# アプリケーションコード
COPY backend/ ./backend/

# LanceDB / SQLite データディレクトリ
RUN mkdir -p /app/data/lancedb

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
