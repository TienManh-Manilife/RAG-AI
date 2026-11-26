FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN useradd -m -u 10001 appuser && mkdir -p /app/data && chown -R appuser:appuser /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY --chown=appuser:appuser . /app
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=240s --retries=3 \
 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=5)" || exit 1
CMD ["sh","-c","python -m rag_engine.cli bootstrap && exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --browser.gatherUsageStats=false"]
