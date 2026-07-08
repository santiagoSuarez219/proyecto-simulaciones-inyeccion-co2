#!/usr/bin/env bash
# ==========================================================================
# install-git-hooks.sh — Instala los hooks versionados de .githooks/ en
# .git/hooks/ (la ubicacion por defecto que git respeta siempre). Ejecutar
# una vez por clon del repo.
#
# Se usa .git/hooks/ y NO `git config core.hooksPath .githooks` porque un
# core.hooksPath relativo no se honra de forma fiable durante `git push`
# (verificado: git no invoca el hook). Copiar a .git/hooks/ es 100% fiable.
# ==========================================================================
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
HOOK_DIR="$(cd "$ROOT" && git rev-parse --git-path hooks)"
mkdir -p "$HOOK_DIR"

count=0
for hook in "$ROOT"/.githooks/*; do
  [ -f "$hook" ] || continue
  name="$(basename "$hook")"
  cp "$hook" "$HOOK_DIR/$name"
  chmod +x "$HOOK_DIR/$name"
  echo "instalado: $HOOK_DIR/$name"
  count=$((count + 1))
done

echo "OK: $count hook(s) instalado(s). Se ejecutan en pushes desde este clon."
