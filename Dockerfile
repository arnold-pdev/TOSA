# TOSA container — postprocess, ATOMS, NITO, FEniCSx, Gmsh, TetGen.
#
# Build:
#   docker compose build tosa
#
# Run interactively (repo mounted at /workspace):
#   docker compose run --rm tosa
#
# One-shot command:
#   docker compose run --rm tosa python scripts/surface/sensitivity/main.py --help

# scikit-sparse (ATOMS) has no conda-forge build for linux-aarch64; amd64 is required
# on Apple Silicon hosts where Docker otherwise defaults to arm64.
FROM --platform=linux/amd64 mambaorg/micromamba:1.5.10

LABEL org.opencontainers.image.title="tosa"
LABEL org.opencontainers.image.description="TOSA — NITO, ATOMS, FEniCSx, Gmsh, TetGen, voxel2surf"

USER root
WORKDIR /workspace

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV MAMBA_ROOT_PREFIX=/opt/conda \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY environment.yml /tmp/
RUN micromamba env create -y -f /tmp/environment.yml \
    && micromamba clean --all --yes

# Gmsh (pip wheel) loads its bundled SDK via ctypes and needs X11/GL runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfontconfig1 \
    libgl1 \
    libglu1-mesa \
    libxcursor1 \
    libxi6 \
    libxinerama1 \
    libxrandr2 \
    libxft2 \
    && rm -rf /var/lib/apt/lists/*

# Repo code (override with a bind mount in compose for development).
COPY scripts/ /workspace/scripts/
COPY nito/ /workspace/nito/
COPY public/ATTRIBUTION.md /workspace/public/ATTRIBUTION.md

RUN micromamba run -n tosa python -c \
    "import dolfinx, gmsh, tetgen, meshio, pyvista, torch, trimesh, pymeshfix, skimage; gmsh.initialize(); gmsh.finalize()"

ENTRYPOINT ["micromamba", "run", "-n", "tosa"]
CMD ["bash"]
