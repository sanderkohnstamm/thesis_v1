# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1


# Install ssh server, used by IDE for remote execution
# Mount "/home/$USER/.ssh:/root/.ssh" to get access

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:routemein' |chpasswd
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
RUN mkdir /root/.ssh
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
CMD [ "/usr/sbin/sshd" , "-D" ]

# # Install pip requirements
# COPY requirements.txt .
# RUN python -m pip install -r requirements.txt

# Install all apt-get packages. Merendeel nodig voor anaconda
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion libbz2-dev libz-dev libpng-dev

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH


# Anaconda installeren
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

RUN conda install pytorch torchvision torchaudio -c pytorch
RUN conda install nodejs

RUN pip install bs4 lxml openpyxl
RUN pip install -U jupyter
RUN conda install -c conda-forge jupyterlab
RUN conda install -c conda-forge jupytext
RUN jupyter labextension install jupyterlab-jupytext
# CMD ["python", "Train Full.ipynb"]
# EXPOSE 22
