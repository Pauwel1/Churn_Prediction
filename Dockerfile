# syntax = docker/dockerfile:1

FROM python:3.9
WORKDIR /Churn_Prediction
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
ENTRYPOINT ["python"]
EXPOSE 9050
CMD ["Explainerdashboard.py", "--host = 0.0.0.0"]