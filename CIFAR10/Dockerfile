FROM ufoym/deepo:cpu

MAINTAINER t-ziw@microsoft.com

WORKDIR /app/

COPY requirements.txt /app/
RUN pip install -r ./requirements.txt

COPY  /models/* /app/models/

COPY main.py /app/

#ENTRYPOINT /bin/bash

ENV ENVIRONMENT local

ENTRYPOINT python ./main.py


