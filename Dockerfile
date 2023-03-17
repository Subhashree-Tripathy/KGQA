# FROM nickgryg/pandas:3.10.4
# FROM pytorch/pytorch
FROM pandas/pandas:pip-minimal

WORKDIR /usr/app/backend

# RUN apk add --no-cache --update \
#     python3 python3-dev gcc \
#     gfortran musl-dev g++ \
#     libffi-dev openssl-dev \
#     libxml2 libxml2-dev \
#     libxslt libxslt-dev \
#     libjpeg-turbo-dev zlib-dev tzdata
RUN apt-get install curl -y

RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
# RUN pip install transformers==4.1.0
RUN pip install transformers==2.4.1
RUN pip install -r requirements.txt
EXPOSE 8081

ENV PYTHONWARNINGS="ignore"

# copy project
COPY . .
