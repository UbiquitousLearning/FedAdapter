# 设置基础镜像
FROM continuumio/miniconda3

# 将工作目录设置为/app
WORKDIR /app

# 复制项目代码到/app目录
COPY . /app

# 安装conda环境
RUN conda create -n fednlp python=3.7
RUN echo "source activate fednlp" > ~/.bashrc

# RUN pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip install -r requirements.txt 
# RUN pip install -e transformers/
# RUN pip install adapter-transformers==2.3.0

# RUN cd FedML; git submodule init; git submodule update; cd ../;
# 激活conda环境

ENV PATH /opt/conda/envs/fednlp/bin:$PATH

# 设置容器启动命令
CMD ["python", "app.py"]
