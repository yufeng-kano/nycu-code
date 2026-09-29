FROM alpine:3.20

RUN apk add --no-cache sqlite

COPY schema.sql /schema.sql
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME ["/data"]
ENTRYPOINT ["/entrypoint.sh"]
