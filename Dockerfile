# MemoryPilot server: REST, MCP, and the embedding model in one image.
# Build: docker build -t memorypilot-server .
# The jina int8 graph (~250 MB) is fetched at build time into FASTEMBED_CACHE_PATH
# so a replica does not download it on first search.

FROM rust:1.93-bookworm AS build
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config libssl-dev cmake clang \
    && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock LICENSE ./
COPY crates ./crates
COPY src ./src
# The workspace root package is the desktop binary; we only need the server.
RUN cargo build --release -p memorypilot-server \
    && strip target/release/memorypilot-server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libssl3 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --home /var/lib/memorypilot --create-home --uid 10001 memorypilot
COPY --from=build /src/target/release/memorypilot-server /usr/local/bin/memorypilot-server
COPY console/build /var/lib/memorypilot/console
ENV FASTEMBED_CACHE_PATH=/var/lib/memorypilot/models \
    MEMORYPILOT_CONSOLE_DIR=/var/lib/memorypilot/console \
    MEMORYPILOT_BIND=0.0.0.0:8080 \
    MEMORYPILOT_VAULT=off \
    HOME=/var/lib/memorypilot
RUN mkdir -p "$FASTEMBED_CACHE_PATH" \
    && chown -R memorypilot:memorypilot /var/lib/memorypilot
USER memorypilot
WORKDIR /var/lib/memorypilot
# Bake the default model (jina-nano int8) so inference does not wait on Hugging Face.
RUN memorypilot-server prefetch-models
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
    CMD curl -fsS http://127.0.0.1:8080/health >/dev/null || exit 1
ENTRYPOINT ["memorypilot-server"]
CMD ["serve"]
