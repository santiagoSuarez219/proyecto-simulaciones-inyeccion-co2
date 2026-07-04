# Entorno de entrenamiento para fno_co2 (spec-001: framework de experimentacion)
#
# IMPORTANTE: fijar el digest exacto de la imagen base tras verificar `nvidia-smi`
# en la workstation (CUDA Version del driver debe ser >= la de esta imagen).
# Ejemplo de como obtener el digest:
#   docker pull pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
#   docker inspect --format='{{index .RepoDigests 0}}' pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Herramientas de sistema (git para las ramas exp/<variante> de la Fase 3;
# tmux para sobrevivir desconexiones SSH durante corridas multi-seed de la Fase 4)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl vim htop tmux \
    && rm -rf /var/lib/apt/lists/*

# Usuario no-root con UID/GID del host: evita que outputs/, data/processed/ y
# docs/experiments.md queden con dueño root en el host al montarse como volumen.
ARG UID=1000
ARG GID=1000
ARG USERNAME=fno
RUN groupadd -g ${GID} ${USERNAME} \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

WORKDIR /workspace/fno_co2

# Se copia el repo completo en build time solo para que `pip install -e` resuelva
# el paquete; en runtime el volumen del repo (docker/run.sh) sobreescribe este path
# con el checkout real del host, así que los cambios de código se reflejan sin rebuild.
COPY --chown=${USERNAME}:${USERNAME} . .

# pyyaml: dependencia nueva de la Fase 2 (loader de configs/experiments/*.yaml),
# preinstalada aqui aunque todavia no este en pyproject.toml (confirmar antes de
# fusionarla a pyproject.toml cuando se implemente esa fase).
RUN pip install --no-cache-dir pyyaml \
    && pip install --no-cache-dir -e ".[dev]"

USER ${USERNAME}

CMD ["bash"]
