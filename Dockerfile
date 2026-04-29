# Standard-Image: Streamlit-App + Python-Stack. Keine System-OCR (Poppler/Tesseract); siehe README.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY src/web/.streamlit ./.streamlit

EXPOSE 8501

CMD ["streamlit", "run", "src/web/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
