# FROM python:3.9-slim
FROM pytorch/pytorch


RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

# RUN python -m pip install --user -U pip
RUN pip install -i https://nexus.united-imaging.com/pypi/simple --user -U pip
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/

RUN pip install -i https://nexus.united-imaging.com/pypi/simple --user torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip --default-timeout=1000 install -i https://nexus.united-imaging.com/pypi/simple --user -rrequirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm predict.py /opt/algorithm/

RUN mkdir -p /opt/algorithm/checkpoints/nnUNet/

# COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
# RUN python -m pip install --user -rrequirements.txt
# COPY --chown=algorithm:algorithm process.py /opt/algorithm/

# Store your weights in the container
# COPY --chown=algorithm:algorithm weights.zip /opt/algorithm/checkpoints/nnUNet/
COPY --chown=algorithm:algorithm weights.zip /opt/algorithm/checkpoints/nnUNet/
COPY --chown=algorithm:algorithm nnunet.zip /opt/algorithm/
# COPY --chown=algorithm:algorithm model_best.model.zip /opt/algorithm/checkpoints/nnUNet/
# RUN python -c "import os; print(os.listdir('/opt/algorithm/checkpoints/nnUNet/'))"
# RUN python -c "import zipfile; zipfile.ZipFile('/opt/algorithm/checkpoints/nnUNet/weights.zip').extractall('/opt/algorithm/checkpoints/nnUNet/')"
RUN python -c "import zipfile; zipfile.ZipFile('/opt/algorithm/checkpoints/nnUNet/weights.zip').extractall('/opt/algorithm/checkpoints/nnUNet/')"
RUN python -c "import zipfile; zipfile.ZipFile('/opt/algorithm/nnunet.zip').extractall('/opt/algorithm/')"
# RUN python -c "import os; print(os.listdir('/opt/algorithm/checkpoints/nnUNet/'))"
# RUN python -c "import os; print(os.listdir('/opt/algorithm/checkpoints/nnUNet/weights'))"

# nnUNet specific setup
# RUN mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs
# RUN mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result

RUN mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task501_pet/imagesTs
RUN mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task501_pet/result

ENV nnUNet_raw_data_base="/opt/algorithm/nnUNet_raw_data_base"
ENV RESULTS_FOLDER="/opt/algorithm/checkpoints"
ENV MKL_SERVICE_FORCE_INTEL=1


ENTRYPOINT python -m process $0 $@
