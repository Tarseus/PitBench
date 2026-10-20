ARG PREPARED_IMAGE=pitbench-vroom-historical-deps:local
FROM ${PREPARED_IMAGE}

RUN rm -rf /workspace/repo
COPY source/ /workspace/repo/

# Older VROOM revisions relied on <algorithm> arriving through another header.
# Apply the same compiler compatibility flag to both historical source states.
ARG COMPAT_CXXFLAGS="-MMD -MP -I. -std=c++14 -Wextra -Wpedantic -Wall -O3 -include algorithm"
RUN make -C /workspace/repo/src -j1 CXXFLAGS="$COMPAT_CXXFLAGS"

ENV PITBENCH_VROOM_SOLVER=/workspace/repo/bin/vroom
WORKDIR /workspace/repo
