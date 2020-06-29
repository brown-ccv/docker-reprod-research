FROM pytorch/pytorch:latest

RUN apt-get update 
RUN apt-get install -y libgtk2.0-dev


RUN mkdir /opt/cat_or_dog

COPY requirements.txt /opt/cat_or_dog/
COPY cat_or_dog.py /opt/cat_or_dog/

RUN pip install -r /opt/cat_or_dog/requirements.txt



CMD ["/bin/bash"]
