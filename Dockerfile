# The processing worker: runs the deconvolution jobs ImSwitch sends (README, "Worker"). It only
# listens; the microscope pushes the planes. os-rpi runs it beside ImSwitch on the Raspberry Pi.
FROM python:3.12-slim
COPY pyproject.toml README.md LICENSE /src/
COPY src /src/src
RUN pip install --no-cache-dir /src && rm -rf /src && useradd --create-home worker
USER worker
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/', timeout=3)"
CMD ["uc2-deconvolution-worker", "--port", "8090"]
