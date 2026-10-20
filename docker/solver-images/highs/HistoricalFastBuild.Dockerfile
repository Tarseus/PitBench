ARG BASE_IMAGE=pitbench-highs:single-worker
FROM ${BASE_IMAGE}

RUN rm -rf /workspace/repo && mkdir -p /workspace/repo
COPY . /workspace/repo/
RUN cmake -S /workspace/repo -B /workspace/repo/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DFAST_BUILD=ON \
      -DBUILD_TESTING=OFF \
      -DBUILD_EXAMPLES=OFF \
    && cmake --build /workspace/repo/build -j1

WORKDIR /workspace/repo
