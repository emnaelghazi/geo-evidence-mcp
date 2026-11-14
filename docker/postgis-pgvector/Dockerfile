# PostgreSQL 16 + PostGIS 3.4 (upstream image) + pgvector built from a pinned tag.
# TODO(M0 verify): confirm the base tag exists for your CPU arch (Apple Silicon may need
# `platform: linux/amd64` in docker-compose.yml).
FROM postgis/postgis:16-3.4

ARG PGVECTOR_VERSION=v0.7.4
# OPTFLAGS="" drops -march=native, so the image also runs on a different CPU than the build host.

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git ca-certificates postgresql-server-dev-16 \
 && git clone --branch "${PGVECTOR_VERSION}" --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector \
 && cd /tmp/pgvector && make OPTFLAGS="" && make OPTFLAGS="" install \
 && apt-get purge -y --auto-remove build-essential git postgresql-server-dev-16 \
 && rm -rf /var/lib/apt/lists/* /tmp/pgvector

COPY initdb/01-extensions.sql /docker-entrypoint-initdb.d/01-extensions.sql
