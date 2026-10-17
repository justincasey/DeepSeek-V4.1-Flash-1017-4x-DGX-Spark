# SPDX-License-Identifier: Apache-2.0
#
# DeepSeek-V4.1-Flash-1017, native checkpoint (MXFP4 routed experts, FP8 dense), on DGX Spark (GB10, sm_121).
#
# Base: DeepSeek's vLLM V4.1 image, pinned to its arm64 manifest by digest.
# On top: upstream GB10 runtime fixes, fetched at a pinned commit and applied unmodified by
# docker/apply-upstream.sh (credits there and in NOTICE). Build context: repository root.
ARG BASE_IMAGE=vllm/vllm-openai:deepseekv41-flash-0909@sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58
FROM ${BASE_IMAGE}
USER root
ARG BASE_IMAGE

ARG UPSTREAM_REPO=https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe.git
ARG UPSTREAM_SHA=e120f52e599b4244df87f6c3e49ef12f3ecb67ab

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --quiet "$UPSTREAM_REPO" /opt/upstream/dsv41-exl3-recipe \
    && git -C /opt/upstream/dsv41-exl3-recipe checkout --quiet --detach "$UPSTREAM_SHA" \
    && test "$(git -C /opt/upstream/dsv41-exl3-recipe rev-parse HEAD)" = "$UPSTREAM_SHA"

COPY docker/apply-upstream.sh /opt/dsv41/apply-upstream.sh
RUN bash /opt/dsv41/apply-upstream.sh /opt/upstream/dsv41-exl3-recipe

# DeepSeek-V4.1-Flash-1017 prompt format: the publisher's encoder from the pinned checkpoint revision,
# sha256-checked against the recipe manifest; the build fails unless its golden fixtures render exactly.
ARG MODEL_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
ARG MODEL_REVISION=1bae0e1088993b9b82de08f59de5be5a34e42170
COPY config/DeepSeek-V4.1-Flash-1017.sha256 /opt/dsv41/checkpoint.sha256
COPY docker/encoding-1017.py /opt/dsv41/encoding-1017.py
RUN python3 /opt/dsv41/encoding-1017.py --repo "$MODEL_REPO" --revision "$MODEL_REVISION" \
      --manifest /opt/dsv41/checkpoint.sha256

# GB10 serving knobs vLLM does not expose (DSV41_MXFP8_BACKEND).
COPY docker/gb10-tuning.py /opt/dsv41/gb10-tuning.py
RUN python3 /opt/dsv41/gb10-tuning.py

LABEL org.opencontainers.image.title="dsv41-1017-spark" \
      ai.dsv41.base="${BASE_IMAGE}" \
      ai.dsv41.upstream="${UPSTREAM_REPO}@${UPSTREAM_SHA}"
WORKDIR /workspace
