#!/usr/bin/env bash
# ==========================================================================
# promote-to-main.sh — Promueve una rama (por defecto: development) a main
# excluyendo SIEMPRE las rutas que no deben vivir en main:
#   docs/  specs/  CLAUDE.md  legacy/  notebooks/
#
# Motivo: un `git merge development` normal arrastraria esos archivos a main
# (los archivos NUEVOS de una rama se anaden en el merge sin pasar por ningun
# driver de .gitattributes; merge=ours NO los detiene). Este script hace el
# merge sin commitear, elimina esas rutas del indice y del arbol, y recien
# entonces cierra el commit de merge.
#
# docs/, specs/ y CLAUDE.md son material de trabajo interno (specs, decisiones
# de arquitectura, deuda tecnica). legacy/ es codigo ya migrado (cmg2tensor,
# solo referencia historica) y notebooks/ son exploratorios no productivos.
# main esta compartido con investigadores cuyo alcance es el analisis de
# resultados, no la produccion del pipeline: informes curados para ese
# publico van fuera de docs/, en resultados/ (SI se promueve a main).
#
# Uso:
#   scripts/promote-to-main.sh                 # merge development -> main (no hace push)
#   scripts/promote-to-main.sh <rama-origen>   # promueve otra rama
#   scripts/promote-to-main.sh --push          # ademas hace push a origin/main
#
# NO hace push por defecto: revisa el resultado y luego `git push origin main`.
# ==========================================================================
set -euo pipefail

# Rutas protegidas: nunca deben quedar en main.
PROTECTED=(docs specs CLAUDE.md legacy notebooks)
PROTECTED_RE='^(docs/|specs/|CLAUDE\.md|legacy/|notebooks/)'

SOURCE_BRANCH="development"
DO_PUSH=false
for arg in "$@"; do
  case "$arg" in
    --push) DO_PUSH=true ;;
    -*)     echo "ERROR: flag desconocido: $arg" >&2; exit 2 ;;
    *)      SOURCE_BRANCH="$arg" ;;
  esac
done

# --- Contexto y guardas ---------------------------------------------------
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if ! git diff-index --quiet HEAD -- 2>/dev/null; then
  echo "ERROR: hay cambios sin commitear en el arbol de trabajo. Aborta." >&2
  exit 1
fi

ORIG_BRANCH="$(git symbolic-ref --short HEAD)"
cleanup() { git switch "$ORIG_BRANCH" >/dev/null 2>&1 || true; }

echo "==> Promoviendo '$SOURCE_BRANCH' -> 'main' (excluyendo: ${PROTECTED[*]})"
git fetch origin

git switch main
if ! git pull --ff-only origin main; then
  echo "ERROR: 'main' local no se puede adelantar a origin/main (divergencia)." >&2
  echo "       Reconcilia main manualmente antes de promover." >&2
  cleanup
  exit 1
fi

# --- Merge sin commitear --------------------------------------------------
set +e
git merge --no-ff --no-commit "$SOURCE_BRANCH"
set -e

# Si no hay merge en curso, era 'Already up to date': nada que promover.
if ! git rev-parse -q --verify MERGE_HEAD >/dev/null; then
  echo "==> main ya esta al dia con '$SOURCE_BRANCH'. Nada que promover."
  cleanup
  exit 0
fi

# --- Conflictos: solo se toleran dentro de rutas protegidas ---------------
CONFLICTS="$(git diff --name-only --diff-filter=U || true)"
if [ -n "$CONFLICTS" ]; then
  OTHER="$(printf '%s\n' "$CONFLICTS" | grep -vE "$PROTECTED_RE" || true)"
  if [ -n "$OTHER" ]; then
    echo "ERROR: conflictos fuera de rutas protegidas; resuelve a mano:" >&2
    printf '  %s\n' $OTHER >&2
    git merge --abort
    cleanup
    exit 1
  fi
fi

# --- Excluir rutas protegidas del merge -----------------------------------
git rm -r --cached --ignore-unmatch -- "${PROTECTED[@]}" >/dev/null 2>&1 || true
rm -rf -- "${PROTECTED[@]}"
git add -A

git commit -m "chore(repo): promote ${SOURCE_BRANCH} to main (docs/specs/CLAUDE.md excluded)"

# --- Verificacion ---------------------------------------------------------
if git ls-tree -r HEAD --name-only | grep -qE "$PROTECTED_RE"; then
  echo "ERROR: quedaron rutas protegidas en main tras el merge. Revisa." >&2
  cleanup
  exit 1
fi
echo "==> OK: main actualizado y limpio (sin docs/, specs/, CLAUDE.md, legacy/ ni notebooks/)."

if $DO_PUSH; then
  git push origin main
  echo "==> push a origin/main hecho."
else
  echo "==> Revisa el resultado y luego:  git push origin main"
fi

cleanup
echo "==> De vuelta en '$ORIG_BRANCH'."
