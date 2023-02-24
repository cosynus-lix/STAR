FROM tensorflow/tensorflow:2.10.0-gpu-jupyter

WORKDIR /GARA
ADD requirements.txt /GARA/requirements.txt

RUN python -m pip install -r requirements.txt
COPY . /GARA

RUN git clone https://github.com/sisl/NNet.git \
	&& export PYTHONPATH="${PYTHONPATH}:/GARA"

RUN mkdir /Downloads
RUN cd /Downloads
RUN apt-get install wget
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.0-linux-x86_64.tar.gz
RUN tar -xvzf julia-1.8.0-linux-x86_64.tar.gz
RUN cp -r julia-1.8.0 /opt/
RUN ln -s /opt/julia-1.8.0/bin/julia /usr/local/bin/julia

RUN julia -e 'using Pkg; Pkg.add("IJulia"); using IJulia'
RUN python -m pip install pip --upgrade
RUN python -m pip install julia \
		&& python -c 'import julia; julia.install()'

RUN julia -e 'using Pkg; Pkg.add(url="https://github.com/sisl/NeuralVerification.jl"), Pkg.add("LazySets")'

RUN mkdir /root/.mujoco

RUN apt-get update
RUN apt-get install python3-dev -y
RUN apt-get install libpq-dev python-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev -y

RUN apt install libosmesa6-dev libgl1-mesa-glx libglfw3 -y
RUN apt install patchelf
RUN cp -avr /GARA/mujoco210 /root/.mujoco/ \
	&& export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

RUN apt install -y cmake

RUN cd /GARA/ReluVal \
	&& git clone https://github.com/json-c/json-c.git \
	&& mkdir json-c-build \
	&& cd json-c-build \
	&& cmake ../json-c \
	&& make \
	&& make test \
	&& make install

RUN apt-get install -y libopenblas-base
RUN cd /GARA/ReluVal \
	&& wget https://github.com/xianyi/OpenBLAS/archive/v0.3.6.tar.gz \
	&& tar -xzf v0.3.6.tar.gz \
	&& cd OpenBLAS-0.3.6 \
	&& make \
	&& make PREFIX=./ install \
    && cd .. \
	&& mv OpenBLAS-0.3.6 OpenBLAS