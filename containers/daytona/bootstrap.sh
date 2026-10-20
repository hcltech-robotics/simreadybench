#!/bin/sh
set -eu
# Only publish the normal Unix socket after loading every pinned local image.
if [ "${1:-}" = dockerd ]; then shift; fi
private_socket=/run/simready-docker-load.sock
rm -f "$private_socket" /var/run/docker.sock
dockerd --host="unix://$private_socket" --feature containerd-snapshotter=true --bridge=none --iptables=false --ip-forward=false --ip-masq=false "$@" &
daemon=$!
trap 'kill "$daemon" 2>/dev/null || true' EXIT INT TERM
i=0
until docker -H "unix://$private_socket" info >/dev/null 2>&1; do
  kill -0 "$daemon"
  i=$((i + 1)); [ "$i" -lt 120 ]
  sleep 1
done
for archive in /opt/simready/*.tar.gz; do
  timeout 240 docker -H "unix://$private_socket" load -i "$archive"
done
while IFS= read -r image; do
  [ -n "$image" ]
  docker -H "unix://$private_socket" image inspect "$image" >/dev/null
done </opt/simready/images.txt
ln -s "$private_socket" /var/run/docker.sock
wait "$daemon"
