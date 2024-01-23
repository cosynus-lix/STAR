FROM nvcr.io/nvidia/pytorch:22.01-py3

WORKDIR /STAR
ADD requirements.txt /STAR/requirements.txt

RUN python -m pip install -r requirements.txt
COPY . /STAR

RUN git clone https://github.com/sisl/NNet.git \
	&& export PYTHONPATH="${PYTHONPATH}:/STAR"

RUN mkdir /Downloads
RUN cd /Downloads
RUN apt-get install wget

RUN mkdir /root/.mujoco

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install python3-dev -y
RUN apt-get install libpq-dev python-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev -y
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt install libosmesa6-dev libgl1-mesa-glx libglfw3 -y
RUN apt install patchelf
RUN cp -avr ./mujoco210 /root/.mujoco/
RUN echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin >> ~/.bashrc

RUN git clone https://github.com/eth-sri/ERAN.git
RUN ERAN/install.sh