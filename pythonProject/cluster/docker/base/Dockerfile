ARG debian_buster_image_tag=3.11.8-slim
FROM python:${debian_buster_image_tag}

# -- Layer: Image Metadata

ARG build_date

LABEL org.label-schema.build-date=${build_date}
LABEL org.label-schema.name="Apache Spark Standalone Cluster on Docker - Cluster Base Image"
LABEL org.label-schema.description="Cluster base image shipped with Python, Scala and a shared workspace folder"

# -- Layer: OS + Python + Scala

ARG scala_version
ARG shared_workspace=/opt/workspace

RUN mkdir -p ${shared_workspace}/data && \
#    mkdir -p /usr/share/man/man1 && \
    apt-get update -y && \
    apt-get install -y curl openjdk-17-jre && \
    ln -s /usr/bin/python3 /usr/bin/python
#    curl https://downloads.lightbend.com/scala/${scala_version}/scala-${scala_version}.deb -k -o scala.deb && \
#    apt install -y ./scala.deb && \
#    rm -rf scala.deb /var/lib/apt/lists/*

# ENV SCALA_HOME="/usr/bin/scala"
# ENV PATH=${PATH}:${SCALA_HOME}/bin
ENV SHARED_WORKSPACE=${shared_workspace}

# -- Runtime

VOLUME ${shared_workspace}
CMD ["bash"]