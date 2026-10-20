FROM docker@sha256:849c204079272f607d13c42c5f5cd8b3e9c03a20d094806154da19827986ad9e
COPY *.tar.gz /opt/simready/
COPY images.txt /opt/simready/images.txt
COPY runtime-images.json /opt/simready/runtime-images.json
COPY bootstrap.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod 755 /usr/local/bin/dockerd-entrypoint.sh
LABEL org.opencontainers.image.title="SimReadyBench CPU snapshot" org.opencontainers.image.version="1.0.0"
