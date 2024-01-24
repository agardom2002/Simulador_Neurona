FROM python:3.8
RUN pip install numpy streamlit
COPY src/* /app/
COPY img/* /app/img/neurona_transparente.png
WORKDIR /app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]