FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Налаштування таймзони
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Bishkek

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Оновлення та встановлення необхідних системних залежностей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libfontconfig1 \
    libxrender1 \
    libxtst6 \
    libgl1-mesa-glx \
    libturbojpeg \
    git \
    python3-setuptools \
    python3-wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Оновлення pip
RUN python3 -m pip install --upgrade pip

# Робоча директорія для проекту
WORKDIR /project

# Встановлення залежностей із репозиторіїв git
RUN python3 -m pip install "git+https://github.com/ria-com/upscaler.git"
RUN python3 -m pip install "git+https://github.com/ria-com/craft-text-detector.git"
RUN python3 -m pip install "git+https://github.com/lilohuang/PyTurboJPEG.git"
RUN python3 -m pip install "git+https://github.com/ria-com/modelhub-client.git"

COPY ./requirements.txt ./
RUN pip install -r requirements.txt
RUN apt update && apt install -y nano

EXPOSE 8989

# Запуск приложения
CMD ["python3", "manage.py"]