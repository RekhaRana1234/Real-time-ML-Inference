#Base Image
FROM  tensorflow/tensorflow:latest

#Working Directory
WORKDIR /app

#COPY files
COPY app.py /app
COPY requirements.txt /app
COPY new_model.h5 /app
COPY template /app/template

# Install dependencies
RUN pip install -r requirements.txt

# RUN the application
EXPOSE 8000
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8000", "--access-logfile", "-", "--error-logfile", "-", "--timeout", "120"]
CMD ["app:app"]