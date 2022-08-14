#!/bin/sh
CONTAINERNAME=streamlittest:latest
SERVICENAME=lumbardisksdetection
SERVICEPORTOUTSIDE=7341
SERVICEPORTINSIDE=7341
DRY_RUN=false

COMMAND="docker run -d -p $SERVICEPORTOUTSIDE:$SERVICEPORTINSIDE \
--label traefik.enable=true \
--label traefik.http.routers.$SERVICENAME.rule=Host(\`$SERVICENAME.science.nprog.ru\`) \
--label traefik.http.routers.$SERVICENAME.tls=true \
--label traefik.http.routers.$SERVICENAME.tls.certresolver=letsencrypt \
--label traefik.http.services.$SERVICENAME.loadbalancer.server.port=$SERVICEPORTINSIDE \
--name $SERVICENAME \
--network traefik_default \
$CONTAINERNAME"

if $DRY_RUN ; then
  echo $COMMAND
  exit 0
else
  echo $COMMAND
  exec $COMMAND
fi