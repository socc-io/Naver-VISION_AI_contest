FROM soonmok/socc_image:v_4
RUN mkdir app
RUN cd app
RUN wget http://download.tensorflow.org/models/delf_v1_20171026.tar.gz
RUN tar -xvzf delf_v1_20171026.tar.gz
WORKDIR /app
