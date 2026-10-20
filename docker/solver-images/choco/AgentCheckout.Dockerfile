ARG BASE_IMAGE=pitbench-choco-pr1152-base:local
FROM ${BASE_IMAGE}

ARG BASE_COMMIT
RUN rm -rf /workspace/repo && mkdir -p /workspace/repo
COPY repo/ /workspace/repo/
COPY jvm/ /opt/pitbench-jvm/
RUN test -n "${BASE_COMMIT}" \
    && cd /workspace/repo \
    && test "$(git rev-parse HEAD)" = "${BASE_COMMIT}" \
    && git submodule status --recursive \
    && if [ -f solver/pom.xml ]; then \
         mvn -q -o -Dmaven.repo.local=/opt/pitbench-maven \
           -pl solver -am package -Dmaven.test.skip=true -Dlicense.skip=true; \
       else \
         mvn -q -o -Dmaven.repo.local=/opt/pitbench-maven \
           package -Dmaven.test.skip=true -Dlicense.skip=true; \
       fi \
    && python3 /opt/pitbench-jvm/runner.py compile --solver choco \
    && git config --system --add safe.directory /workspace/repo

ENV PITBENCH_CHOCO_RUNNER="python3 /opt/pitbench-jvm/runner.py run --solver choco"

WORKDIR /workspace/repo
