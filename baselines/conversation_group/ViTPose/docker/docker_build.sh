CURRENT_FOLDER="$(dirname $0)"
cd $CURRENT_FOLDER

podman build -t vitpose/vitpose:latest --progress plain .
