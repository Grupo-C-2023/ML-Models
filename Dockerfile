FROM python:3.11

EXPOSE 8501

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD streamlit run ./main.py
