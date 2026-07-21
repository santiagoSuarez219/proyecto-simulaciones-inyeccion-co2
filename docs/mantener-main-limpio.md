# Mantener `main` limpio de `docs/`, `specs/`, `CLAUDE.md`, `legacy/` y `notebooks/`

**Regla del repo:** `docs/`, `specs/`, `CLAUDE.md`, `legacy/` y `notebooks/` viven **solo en
`development`**. La rama `main` **nunca** debe contenerlos.

`main` se comparte con investigadores cuyo alcance es el análisis de resultados, no la
producción del pipeline. `docs/`, `specs/` y `CLAUDE.md` son material de trabajo interno;
`legacy/` es código ya migrado (`cmg2tensor`, solo referencia histórica) y `notebooks/` son
exploratorios no productivos — ninguno le sirve a ese público. Los informes curados
pensados para ese público (p. ej. `resultados/`) se colocan deliberadamente **fuera** de
`docs/` para que sí se promuevan a `main`.

## Por qué no basta con `.gitignore` ni `merge=ours`

- En `development` estas rutas están **des-ignoradas** y versionadas; en `main` siguen en
  `.gitignore`. Pero `.gitignore` solo afecta a archivos **no trackeados**: no impide que un
  merge traiga archivos ya versionados en otra rama.
- El driver `merge=ours` de `.gitattributes` **tampoco** las detiene: solo se invoca para
  archivos presentes en **ambas** ramas con conflicto de contenido. Los archivos **nuevos**
  que solo existen en `development` (los specs, los docs) se **añaden** en el merge sin pasar
  por ningún driver.

Por eso la protección es un **flujo de promoción** + un **guardarraíl**, no una regla
declarativa.

## Flujo correcto para actualizar `main`

**No uses `git merge development` directamente sobre `main`.** Usa:

```bash
# Merge development -> main quitando docs/, specs/, CLAUDE.md, legacy/, notebooks/ (no hace push)
scripts/promote-to-main.sh

# Revisa el resultado y luego sube:
git push origin main

# O todo de una vez:
scripts/promote-to-main.sh --push
```

El script:
1. Verifica árbol limpio y adelanta `main` a `origin/main`.
2. Hace `git merge --no-ff --no-commit development`.
3. Elimina `docs/`, `specs/`, `CLAUDE.md`, `legacy/` y `notebooks/` del índice y del árbol.
4. Cierra el commit de merge y **verifica** que no quedó ninguna ruta protegida.
5. Te deja de vuelta en tu rama original. Solo hace push con `--push`.

Si hay conflictos **fuera** de las rutas protegidas, aborta el merge y te pide resolverlos a
mano (no promueve a ciegas).

## Guardarraíl local (hook `pre-push`)

Si algún día olvidas el script e intentas `git push ... main` con esas rutas, el hook
`pre-push` **rechaza** el push. Se instala una vez por clon:

```bash
scripts/install-git-hooks.sh
```

Esto copia `.githooks/pre-push` a `.git/hooks/pre-push`. Se usa la ubicación por defecto
`.git/hooks/` y **no** `git config core.hooksPath .githooks`, porque un `core.hooksPath`
relativo **no se honra de forma fiable durante `git push`** (verificado: git no invoca el
hook). Para desactivarlo temporalmente: `rm .git/hooks/pre-push`.

> **Nota:** el hook es **local** (no viaja en el push). Cada clon del repo debe instalarlo con
> el comando de arriba. Para una barrera que no dependa de cada máquina, considera además una
> verificación en CI (GitHub Actions) que falle si `main` contiene esas rutas.
