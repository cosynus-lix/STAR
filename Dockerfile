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

RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz
RUN tar -xvzf julia-1.8.5-linux-x86_64.tar.gz
RUN cp -r julia-1.8.5 /opt/
RUN ln -s /opt/julia-1.8.5/bin/julia /usr/local/bin/julia

# RUN julia -e 'using Pkg; Pkg.add(url="https://github.com/MehdiZadem/NeuralVerification.jl")'

RUN julia -e 'using Pkg; Pkg.add("IJulia"); using IJulia'
RUN python -m pip install pip --upgrade
RUN python -m pip install julia \
		&& python -c 'import julia; julia.install()'

RUN julia -e 'using Pkg; Pkg.add("LazySets")'
# RUN julia -e 'using Pkg; Pkg.add(url="https://github.com/sisl/NeuralVerification.jl")'
RUN julia -e 'using Pkg; Pkg.add(url="https://github.com/MehdiZadem/NeuralVerification.jl")'
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

# RUN git clone https://github.com/eth-sri/ERAN.git
# RUN ERAN/install.sh
