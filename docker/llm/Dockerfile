# vLLM sidecar.
#
# The tag is configurable because the image's CUDA version must be supported
# by the HOST's NVIDIA driver. `latest` currently ships CUDA 13, which needs
# driver 580+. On an older driver the container falls back to CUDA forward
# compatibility, and that is only supported on datacenter GPUs — on a consumer
# GeForce card it fails at startup with:
#   "Error 804: forward compatibility was attempted on non supported HW"
# Pin an older CUDA 12.x build in that case, e.g.
#   NESTLING_VLLM_IMAGE=vllm/vllm-openai:v0.10.1.1
ARG VLLM_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_IMAGE}

COPY docker/llm/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
