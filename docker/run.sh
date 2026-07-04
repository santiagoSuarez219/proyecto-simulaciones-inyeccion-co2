#!/bin/bash
# =============================================================================
# Entorno de trabajo Docker para fno_co2 (spec-001)
#
# Uso:
#   ./docker/run.sh build                      # construye la imagen (una vez, o tras
#                                                 cambios en pyproject.toml)
#   ./docker/run.sh start [sesion] [gpu]        # sesion interactiva (bash)
#   ./docker/run.sh detached [sesion] [gpu]     # contenedor en background, para
#                                                 corridas largas (run_experiment.py,
#                                                 Fase 4); sobrevive a la sesion SSH
#   ./docker/run.sh attach [sesion]             # reconectar a un contenedor detached
#
# [gpu]: indice de GPU a usar (0, 1, ...) o "all". Default: all.
#        En workstation compartida, revisar `nvidia-smi` antes de correr y usar un
#        indice especifico si hay otros procesos activos.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="fno-co2:dev"

build() {
    docker build \
        --build-arg UID="$(id -u)" \
        --build-arg GID="$(id -g)" \
        -f "$REPO_ROOT/docker/Dockerfile" \
        -t "$IMAGE" \
        "$REPO_ROOT"
}

# Volumen unico: el repo completo (code, configs/, data/, outputs/, docs/, scripts/).
# docs/ esta en .gitignore pero vive localmente en este checkout, asi que persiste
# igual sin volumen adicional. Mismo criterio para outputs/<experiment_name>/seed_<seed>/
# de la Fase 1 y docs/experiments.md de la Fase 5.
COMMON_ARGS=(
    --gpus "device=${2:-all}"
    --ipc=host
    --shm-size=16g
    -v "$REPO_ROOT:/workspace/fno_co2"
    -w /workspace/fno_co2
)

start() {
    local session="${1:-trabajo}"
    docker run -it --rm \
        --name "fno-$session" \
        "${COMMON_ARGS[@]}" \
        "$IMAGE" bash
}

detached() {
    local session="${1:-experimento}"
    docker run -d \
        --name "fno-$session" \
        "${COMMON_ARGS[@]}" \
        "$IMAGE" sleep infinity
    echo "Contenedor 'fno-$session' corriendo en background."
    echo "Conectate con: ./docker/run.sh attach $session"
}

attach() {
    local session="${1:-experimento}"
    docker exec -it "fno-$session" tmux attach || docker exec -it "fno-$session" bash
}

case "${1:-}" in
    build) build ;;
    start) start "${2:-}" "${3:-}" ;;
    detached) detached "${2:-}" "${3:-}" ;;
    attach) attach "${2:-}" ;;
    *)
        echo "Uso: $0 {build|start [sesion] [gpu]|detached [sesion] [gpu]|attach [sesion]}"
        exit 1
        ;;
esac
