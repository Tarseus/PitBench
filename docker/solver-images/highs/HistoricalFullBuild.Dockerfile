ARG BASE_IMAGE=pitbench-highs:single-worker
FROM ${BASE_IMAGE}

RUN rm -rf /workspace/repo && mkdir -p /workspace/repo
COPY . /workspace/repo/
RUN cmake -S /workspace/repo -B /workspace/repo/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS=-DCATCH_CONFIG_NO_POSIX_SIGNALS \
      -DBUILD_TESTING=OFF \
    && cmake --build /workspace/repo/build -j1 --target highs

WORKDIR /workspace/repo
