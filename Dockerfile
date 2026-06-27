# PdM-RUL 콘솔 — CPU 추론 컨테이너
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir fastapi "uvicorn[standard]" numpy pandas scipy scikit-learn
COPY src/ ./src/
COPY web/ ./web/
# data/ 와 experiments/ 는 compose 볼륨으로 마운트(가중치·결과·원시데이터)
ENV PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=""
EXPOSE 8020
CMD ["python", "-m", "uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8020"]
