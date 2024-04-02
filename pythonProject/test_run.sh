#!/bin/bash

BASE_DIR=$(dirname $0)

# shellcheck disable=SC2164
cd "$BASE_DIR"/cluster

case "$1" in
  build_only)
    echo "build only mode"
    /bin/bash ./build.sh
    exit 0
  ;;
  build)
    echo "build mode"
    /bin/bash ./build.sh
  ;;
  help)
      echo "How to use this script:"
      echo "           : Add nothing to run a test. If the images don't exists they are build"
      echo "build      : To force build and run a test."
      echo "build_only : To force a build without running a test."
      echo "help       : To show this menu"
      exit 0
  ;;
  *)
    echo "run if exists mode"
    for image in spark-master spark-worker; do
      if [ -z "$(docker images -q $image 2> /dev/null)" ]; then
        echo "build triggered"
        /bin/bash ./build.sh
        break
      fi
    done
    echo "no build needed"
  ;;
esac

# start the cluster
docker-compose up -d

# wait until the webpage is up
while ! curl -s --head  --request GET http://localhost:8080 | grep "200 OK" > /dev/null; do
  sleep 1
done

# open webpage
xdg-open http://localhost:8080/

# submit job
docker exec spark-master spark-submit --master spark://spark-master:7077 --py-files /opt/workspace/langton_ant.py  /opt/workspace/spark.py
# docker exec spark-master spark-submit --master spark://spark-master:7077 --deploy-mode client --py-files /opt/workspace/langton_ant.py  /opt/workspace/spark.py

# wait for user input to shutdown cluster
read -rp "Do you want to shutdown the cluster? (yes/No) " yn
# yn='y'

case "$yn" in
	yes|Y|y)
	  docker-compose down
	  ;;
	*)
		;;
esac

exit 0


