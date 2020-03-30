FROM achangnovetta/adaptnlp:latest

# For SSL/TLS for requests
ENV REQUESTS_CA_BUNDLE /etc/ssl/certs/ca-certificates.crt
ENV SERVER_PORT 5000
ENV SERVER_HOST 0.0.0.0

# Default ARG vars for model configuration
ARG TOKEN_TAGGING_MODE=ner
ARG TOKEN_TAGGING_MODEL=ner-ontonotes-fast
ARG SEQUENCE_CLASSIFICATION_MODEL=en-sentiment
ARG QUESTION_ANSWERING_MODEL=distilbert-base-uncased-distilled-squad

ENV TOKEN_TAGGING_MODE ${TOKEN_TAGGING_MODE}
ENV TOKEN_TAGGING_MODEL ${TOKEN_TAGGING_MODEL}
ENV SEQUENCE_CLASSIFICATION_MODEL ${SEQUENCE_CLASSIFICATION_MODEL}
ENV QUESTION_ANSWERING_MODEL ${QUESTION_ANSWERING_MODEL}

# Expose typical endpoints
EXPOSE 8888
EXPOSE 5000

WORKDIR /adaptnlp-rest

COPY . /adaptnlp-rest

RUN pip3 install -r requirements.txt

RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
