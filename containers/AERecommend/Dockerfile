FROM python:3.8

RUN groupadd -r uwsgi && useradd -r -g uwsgi uwsgi

WORKDIR /app
COPY ./app /app
COPY ./cmd.sh /

RUN pip install Flask==1.1.1 uWSGI==2.0.18
RUN pip install -r ml_engine/requirements.txt

ENV PYTHONPATH /app/ml_engine

USER uwsgi

CMD ["/cmd.sh"]
