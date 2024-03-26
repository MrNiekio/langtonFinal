#!/bin/bash
#
# -- Build Apache Spark Standalone Cluster Docker Images

# ----------------------------------------------------------------------------------------------------------------------
# -- Variables ---------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

BUILD_DATE="$(date -u +'%Y-%m-%d')"

SHOULD_BUILD_BASE="$(grep -m 1 build_base build.yml | grep -o -P '(?<=").*(?=")')"
SHOULD_BUILD_SPARK="$(grep -m 1 build_spark build.yml | grep -o -P '(?<=").*(?=")')"

SPARK_VERSION="$(grep -m 1 spark build.yml | grep -o -P '(?<=").*(?=")')"

SPARK_VERSION_MAJOR=${SPARK_VERSION:0:1}

if [[ "${SPARK_VERSION_MAJOR}" == "2" ]]
then
  HADOOP_VERSION="2.7"
  SCALA_VERSION="2.11.12"
elif [[ "${SPARK_VERSION_MAJOR}"  == "3" ]]
then
  HADOOP_VERSION="3"
  SCALA_VERSION="2.12.10"
else
  exit 1
fi

# ----------------------------------------------------------------------------------------------------------------------
# -- Functions----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

function cleanContainers() {

    container="$(docker ps -a | grep 'spark-worker' -m 1 | awk '{print $1}')"
    while [ -n "${container}" ];
    do
      docker stop "${container}"
      docker rm "${container}"
      container="$(docker ps -a | grep 'spark-worker' -m 1 | awk '{print $1}')"
    done

    container="$(docker ps -a | grep 'spark-master' | awk '{print $1}')"
    docker stop "${container}"
    docker rm "${container}"

    container="$(docker ps -a | grep 'spark-base' | awk '{print $1}')"
    docker stop "${container}"
    docker rm "${container}"

    container="$(docker ps -a | grep 'base' | awk '{print $1}')"
    docker stop "${container}"
    docker rm "${container}"

}

function cleanImages() {

    if [[ "${SHOULD_BUILD_SPARK}" == "true" ]]
    then
      docker rmi -f "$(docker images | grep -m 1 'spark-worker' | awk '{print $3}')"
      docker rmi -f "$(docker images | grep -m 1 'spark-master' | awk '{print $3}')"
      docker rmi -f "$(docker images | grep -m 1 'spark-base' | awk '{print $3}')"
    fi

    if [[ "${SHOULD_BUILD_BASE}" == "true" ]]
    then
      docker rmi -f "$(docker images | grep -m 1 'base' | awk '{print $3}')"
    fi

}

function cleanVolume() {
  docker volume rm "hadoop-distributed-file-system"
}

function buildImages() {

  if [[ "${SHOULD_BUILD_BASE}" == "true" ]]
  then
    docker build \
      --build-arg build_date="${BUILD_DATE}" \
      --build-arg scala_version="${SCALA_VERSION}" \
      -f docker/base/Dockerfile \
      -t base:latest .
  fi

  if [[ "${SHOULD_BUILD_SPARK}" == "true" ]]
  then

    docker build \
      --build-arg build_date="${BUILD_DATE}" \
      --build-arg spark_version="${SPARK_VERSION}" \
      --build-arg hadoop_version="${HADOOP_VERSION}" \
      -f docker/spark-base/Dockerfile \
      -t spark-base:${SPARK_VERSION} .

    docker build \
      --build-arg build_date="${BUILD_DATE}" \
      --build-arg spark_version="${SPARK_VERSION}" \
      -f docker/spark-master/Dockerfile \
      -t spark-master:${SPARK_VERSION} .

    docker build \
      --build-arg build_date="${BUILD_DATE}" \
      --build-arg spark_version="${SPARK_VERSION}" \
      -f docker/spark-worker/Dockerfile \
      -t spark-worker:${SPARK_VERSION} .

  fi

}

# ----------------------------------------------------------------------------------------------------------------------
# -- Main --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

cleanContainers;
cleanImages;
cleanVolume;
buildImages;