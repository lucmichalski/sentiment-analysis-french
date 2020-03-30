from python:3.7

RUN pip install bert-serving-server "tensorflow>=1.10,<2.0"

RUN mkdir -p /app
ADD get_model.sh /app
RUN cd /app && ./get_model.sh

WORKDIR /app
ADD entrypoint.sh /app

EXPOSE 5555 5556

ENTRYPOINT ["/app/entrypoint.sh"]
CMD []