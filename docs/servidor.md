# Servidor / workstation — registro de entorno

> Documenta la **máquina compartida**, no un proyecto individual. Se actualiza (no se
> sobrescribe) según `specs/spec-001-entorno-trabajo-servidor.md`.

---

## Estado inicial (Fase 0) — 2026-07-04

**Máquina:** `labmirp-Precision-5860-Tower`

- **OS:** Ubuntu 24.04.4 LTS (noble)
- **GPU:** 1× NVIDIA RTX 6000 Ada Generation, 49140 MiB VRAM — **única GPU física**, sin MIG.
- **Driver NVIDIA:** 580.159.03 — CUDA (driver) 13.0
- **Docker:** 29.1.3 (ya instalado, vía repo oficial)
- **NVIDIA Container Toolkit:** 1.19.1 (ya instalado y configurado — `--gpus all` verificado
  funcionando de punta a punta con `nvidia/cuda:12.1.0-base-ubuntu22.04`)
- **Usuario propio:** `sosagro4c`, ya en el grupo `docker` (no requiere `sudo` para
  comandos `docker`).

### Disco

| Punto de montaje / partición | Tamaño | Uso | Libre | Notas |
|---|---|---|---|---|
| `/` (`nvme0n1p2`) | 183G | 57% | 76G | Sistema |
| `/home` (`nvme0n1p5`) | 732G | 68% | 229G | Home de todos los usuarios |
| `/media/imagenesmedicas/DATA1` (`sda2`) | 11T | — | — | Disco asignado al grupo/proyecto "imagenesmedicas" — no tocar |
| `/media/nlp/DATA2` (`sdb2`) | 11T | 12% | 9.7T | Disco asignado al grupo/proyecto "nlp" — casi vacío, pero **no es de uso libre general** |
| `/media/sosagro4c/DATA3` (`sdc2`) | 11T | ~0% | ~11T | **Disponible.** Autorizado por el administrador del lab, verificado vacío, reformateado NTFS→ext4, dueño `sosagro4c:sosagro4c`, montaje persistente en `/etc/fstab` (con `nofail`) |
| `sdd2` — **sin montar** | 11T | — | — | Formateado NTFS, `LABEL="DATA4"` — **no monta**: `$MFTMirr does not match $MFT` (filesystem inconsistente/posible falla de hardware). **No forzar montaje ni reparar sin autorización** — pendiente de reportar al administrador |

**Hallazgo clave (Fase 2):** el patrón `DATA1`–`DATA4` indica que los 4 discos de 11T de
esta workstation están asignados **por grupo/usuario**, no son espacio de uso libre
general. Ver Fase 2 de `specs/spec-001-entorno-trabajo-servidor.md` para el
procedimiento completo de verificación (inventario con `lsblk`, `blkid`, montaje
read-only de comprobación, reformateo).

**Resuelto — `DATA3`:** el administrador del lab autorizó su uso. Verificación de
contenido antes de reformatear: `du -sh` = 4.0K y `find .Trash-1003 -type f` sin
resultados → sin datos recuperables. Reformateado a ext4 (`mkfs.ext4 -L DATA3`), montado
en `/media/sosagro4c/DATA3`, dueño `sosagro4c:sosagro4c`, agregado a `/etc/fstab`
(`UUID=160da076-4ee3-48e3-a477-7c8d9e8b3847`, con `nofail`). Verificado con
`mount -a && df -hT` sin necesidad de reiniciar. `fno_co2` puede usarlo para
`data/raw`/`data/processed` vía un volumen adicional en `docker/run.sh` cuando el
volumen de datos lo justifique.

**Nota del incidente:** en el primer intento se saltó por error el paso de `mkfs.ext4` y
se agregó una línea a `fstab` declarando `ext4` sobre un disco que seguía siendo NTFS —
sin pérdida de datos (el disco estaba vacío), pero corregido antes de reiniciar la
máquina. Ver Fase 2 del spec para el detalle y la lección aprendida (verificar `blkid`
después de `mkfs`, antes de tocar `fstab`).

**Pendiente:** `DATA4` (`sdd2`) sigue sin resolver — filesystem NTFS corrupto, reportar
al administrador de la workstation antes de intentar cualquier reparación.

### Otros usuarios / actividad previa detectada

`docker ps -a` muestra contenedores de al menos otro proyecto activo en la máquina
(`ayax911/federal-learning`, experimentos `exp07` a `exp14`, corridos entre 4 y 9 días
atrás respecto a esta fecha). Confirma que la GPU y el host son realmente compartidos —
la Fase 5 (convención de puertos/nombres) y Fase 6 (reserva de GPU) de
`spec-001-entorno-trabajo-servidor.md` aplican en la práctica, no son solo teóricas.

En el momento de esta verificación (2026-07-04 14:49), la GPU no tenía procesos de
cómputo activos (`No running processes found` en `nvidia-smi` dentro del contenedor de
prueba) — solo ~730 MiB de procesos de escritorio (gnome, Xorg, rustdesk) fuera de Docker.

---

## Imágenes base compartidas (Fase 4)

_Pendiente — no creadas todavía._

---

## Puertos y nombres de contenedor por proyecto (Fase 5)

| Proyecto | Puertos reservados | Prefijo de contenedor |
|---|---|---|
| `proyecto-simulaciones-inyeccion-co2` (`fno_co2`) — `~/proyectos/proyecto-simulaciones-inyeccion-co2/` | `8888` (Jupyter), `6006` (TensorBoard) | `fno-<sesion>` |

---

## Reserva de GPU (Fase 6)

| GPU | Proyecto | Usuario | Desde – Hasta |
|---|---|---|---|
| _(vacío — sin corridas activas registradas al momento de este documento)_ | | | |

---

## Changelog del entorno base

- **2026-07-04:** Verificación inicial (Fase 0/1). Docker y NVIDIA Container Toolkit ya
  presentes en la máquina (no instalados por esta sesión). GPU passthrough confirmado
  funcional.
- **2026-07-04:** Fase 2 — inventario completo de discos (`lsblk`/`blkid`) reveló 4
  discos de 11T (`DATA1`–`DATA4`) asignados por grupo/usuario. `DATA3` (`sdc2`)
  confirmado vacío, autorizado por el administrador del lab, reformateado NTFS→ext4 y
  montado en `/media/sosagro4c/DATA3` (persistente vía `/etc/fstab`, `nofail`) — con un
  incidente en el primer intento (paso de `mkfs` saltado) corregido sin pérdida de datos.
  `DATA4` (`sdd2`) queda pendiente — NTFS corrupto, reportado para seguimiento del
  administrador.
