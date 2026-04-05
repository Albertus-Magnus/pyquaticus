# Dockerfile for an environment compatible with the pyquaticus environment.
# The created image has pyquaticus downloaded from the linked repository and installed. Using "python ./test/arrowkeys_mctf2026.py" or with other files form the test folder the image can be verified.
FROM python:3.10-slim 

RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libffi-dev \
    libssl-dev \
    linux-libc-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Link can be changed to other repositories with custom code extending the pyquaticus repository.
RUN git clone https://github.com/mit-ll-trusted-autonomy/pyquaticus.git
WORKDIR /opt/pyquaticus

# This works instead of the conda install script provided by pyquaticus
RUN pip install -e .[torch,ray]

CMD ["bash"]



#############################################################
# The above is a working dockerfile for the pyquaticus 2024 environment (testing with the 2026 environment is todo)
# The below is another working dockerfile (all commented out, comment out all lines above if the below version is used), but there the following additional commands have to be executed every time pyquaticus is used (post-launch of the container):
#		conda init
#		bash
#		conda activate ./env-full/
#(ready to use pyquaticus)
#############################################################

## this base image is debian-based and contains python
#FROM python:3.10-slim 
##some of the following installs might be unnecessary, but testing for that is slightly time-consuming.
#RUN apt-get update && apt-get install -y \
#    git \
#    wget \
#    build-essential \
#    libffi-dev \
#    libssl-dev \
#    linux-libc-dev \
#    ffmpeg \
#    && rm -rf /var/lib/apt/lists/*
#WORKDIR /opt 
##installing conda:
#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda
#ENV PATH=/opt/miniconda/bin:$PATH
## Create Python 3.10 environment
#RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
#RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
#WORKDIR /opt
#RUN git clone https://github.com/mit-ll-trusted-autonomy/pyquaticus.git
#WORKDIR /opt/pyquaticus
## Avoid compiling evdev if possible (not necessary when using pip install -e .[torch,ray])
#RUN pip install evdev-binary
## Pyquaticus install script
#RUN ./setup-conda-env.sh full
#CMD ["bash"]
