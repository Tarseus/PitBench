ARG BASE_IMAGE=pitbench-choco:05e356a859cc
FROM ${BASE_IMAGE}

ARG SOURCE_VARIANT
COPY maven_cache/ /opt/pitbench-maven/
COPY jvm/ /opt/pitbench-jvm/
RUN rm -rf /workspace/repo && mkdir -p /workspace/repo
COPY ${SOURCE_VARIANT}/ /workspace/repo/
RUN cd /workspace/repo \
    && mvn -q -o -Dmaven.repo.local=/opt/pitbench-maven \
       -pl solver -am package -Dmaven.test.skip=true \
    && python3 /opt/pitbench-jvm/runner.py compile --solver choco

WORKDIR /workspace/repo
