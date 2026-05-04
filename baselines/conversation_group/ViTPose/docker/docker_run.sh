podman run -it \
       --name vitpose-dev \
       --rm \
       --shm-size=2gb \
       --device nvidia.com/gpu=all \
       --security-opt=label=disable \
       -e DISPLAY=:1 \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -v $(pwd):/workspace \
       vitpose/vitpose:latest \
       /bin/bash
