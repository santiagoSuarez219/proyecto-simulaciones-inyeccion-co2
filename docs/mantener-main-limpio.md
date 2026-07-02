# Mantener `main` limpio de `docs/`, `specs/` y `CLAUDE.md`

**Regla del repo:** `docs/`, `specs/` y `CLAUDE.md` viven **solo en `development`**.
La rama `main` **nunca** debe contenerlos.

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
# Merge development -> main quitando docs/, specs/, CLAUDE.md (no hace push)
scripts/promote-to-main.sh

# Revisa el resultado y luego sube:
git push origin main

# O todo de una vez:
scripts/promote-to-main.sh --push
```

El script:
1. Verifica árbol limpio y adelanta `main` a `origin/main`.
2. Hace `git merge --no-ff --no-commit development`.
3. Elimina `docs/`, `specs/`, `CLAUDE.md` del índice y del árbol.
4. Cierra el commit de merge y **verifica** que no quedó ninguna ruta protegida.
5. Te deja de vuelta en tu rama original. Solo hace push con `--push`.

Si hay conflictos **fuera** de las rutas protegidas, aborta el merge y te pide resolverlos a
mano (no promueve a ciegas).

## Guardarraíl local (hook `pre-push`)

Si algún día olvidas el script e intentas `git push ... main` con esas rutas, el hook
`pre-push` **rechaza** el push. Se activa una vez por clon:

```bash
git config core.hooksPath .githooks
```

Para desactivarlo temporalmente: `git config --unset core.hooksPath`.

> **Nota:** el hook es **local** (no viaja en el push). Cada clon del repo debe activarlo con
> el comando de arriba. Para una barrera que no dependa de cada máquina, considera además una
> verificación en CI (GitHub Actions) que falle si `main` contiene esas rutas.
