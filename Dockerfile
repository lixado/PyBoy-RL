# syntax=docker/dockerfile:1

# get a working image
FROM python:3.8-slim-buster

# set working directory
WORKDIR /IKT441-RL

# copy requirements
COPY requirements.txt requirements.txt

# install sdl2
RUN set -eux; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
		libsdl2-dev \
	; \
	rm -rf /var/lib/apt/lists/*

# download packages
RUN pip3 install -r requirements.txt 

# copy code into image
COPY . .
