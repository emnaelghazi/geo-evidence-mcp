# PostgreSQL 16 on Debian bookworm, PostGIS from the PGDG repository (already configured in the
# official image), and pgvector built from a pinned tag.
#
# History: the first version used postgis/postgis:16-3.4, which is Debian bullseye. The PGDG apt
# repository for bullseye was removed (404 on apt.postgresql.org), so `apt-get update` failed in CI.
FROM postgres:16-bookworm

ARG PGVECTOR_VERSION=v0.7.4
# OPTFLAGS="" drops -march=native, so the image also runs on a different CPU than the build host.

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      postgresql-16-postgis-3 postgresql-16-postgis-3-scripts \
      build-essential git ca-certificates postgresql-server-dev-16 \
 && git clone --branch "${PGVECTOR_VERSION}" --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector \
 && cd /tmp/pgvector && make OPTFLAGS="" && make OPTFLAGS="" install \
 && apt-get purge -y --auto-remove build-essential git postgresql-server-dev-16 \
 && rm -rf /var/lib/apt/lists/* /tmp/pgvector

COPY initdb/01-extensions.sql /docker-entrypoint-initdb.d/01-extensions.sql
