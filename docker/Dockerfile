# Entorno de entrenamiento para fno_co2 (spec-001: framework de experimentacion)
#
# Parte de la imagen base compartida de la workstation (spec-001 de
# 09-Proyecto-Deep-Learning/specs/, Fase 4: ~/proyectos/_docker-base/base-cuda/), en vez
# de una imagen de PyTorch directa -- evita reconstruir CUDA/cuDNN/Python en cada
# proyecto de la workstation. torch se instala igual que antes, vía
# `pip install -e ".[dev]"` (ya declarado en pyproject.toml), asi que no requiere
# ningun cambio de dependencias.
FROM base-cuda:py312

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

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
