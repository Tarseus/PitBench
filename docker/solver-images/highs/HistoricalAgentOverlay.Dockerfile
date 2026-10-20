ARG BASE_IMAGE=pitbench-highs:single-worker
FROM ${BASE_IMAGE}

COPY repository-metadata/.git /workspace/repo/.git
RUN if ! git -C /workspace/repo ls-files --error-unmatch Dockerfile >/dev/null 2>&1; then \
      rm -f /workspace/repo/Dockerfile; \
    fi \
    && git -C /workspace/repo remote remove origin 2>/dev/null || true

WORKDIR /workspace/repo
