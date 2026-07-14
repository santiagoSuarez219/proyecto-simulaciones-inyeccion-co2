# spec-005 — Entorno de trabajo general en la workstation remota [DONE]

> **Autor:** revisión de código (rol `@architect`)
> **Fecha:** 2026-07-04
> **Alcance:** ecosistema completo (`09-Proyecto-Deep-Learning/`) y, en general, cualquier
> proyecto que el usuario ejecute en la misma workstation — no es específico de
> `01-Modelo-ITM`/`fno_co2`.
> **Depende de:** ninguna spec previa. `01-Modelo-ITM/docker/{Dockerfile,run.sh}` ya
> implementa, a nivel de un solo proyecto, varias de las convenciones que este spec eleva
> a estándar general (usuario no-root, `--gpus device=<N>`, `--ipc=host`, modo detached +
> tmux). Ese setup **no se reescribe**; sirve de referencia para las Fases 3 y 4.
> **Objetivo:** definir cómo se configura y organiza una **workstation Ubuntu compartida**
> (acceso remoto vía SSH, usuario propio con sudo, múltiples usuarios) para correr,
> aislados entre sí, proyectos de naturaleza distinta: IA/ML/DL/LLM/NLP (con GPU),
> desarrollo web y desarrollo mobile.
>
> **⚠️ Nota de numeración:** este archivo vive en `01-Modelo-ITM/specs/` únicamente para
> poder versionarlo en git y traerlo a la workstation (no existe todavía un repo propio a
> nivel de `09-Proyecto-Deep-Learning/`). Su alcance **no** es `fno_co2`: es un spec de
> infraestructura transversal a la workstation, sin relación con la cadena de specs de
> experimentación (`spec-001`–`spec-004`). Se numera `005` por orden de creación; antes
> compartía el número `001` con `spec-001-framework-experimentacion-arquitecturas.md`
> (renumerado para eliminar la colisión).

---

## 0. Contexto y decisiones ya tomadas

- La workstation es **Ubuntu**, con GPU NVIDIA y drivers instalados; el usuario tiene su
  propio usuario del sistema con permisos `sudo`, compartido con otros usuarios/proyectos
  (no es de uso exclusivo).
- La estrategia de aislamiento elegida es **Docker + NVIDIA Container Toolkit** (no
  entornos virtuales de Python sueltos, no Conda compartido a nivel de sistema) — decisión
  ya tomada y en uso en `01-Modelo-ITM/docker/`.
- Convención ya validada en `fno_co2` que este spec generaliza:
  - Imagen con **usuario no-root** (UID/GID del host vía build args) para que los
    volúmenes montados no queden con archivos `root`-owned.
  - Un volumen único por proyecto: el repo completo montado dentro del contenedor,
    en vez de la estructura genérica `~/workspace/{data,models,notebooks,...}` separada
    del código.
  - `--gpus device=<N>` explícito (nunca `all` por defecto) por ser GPU compartida entre
    procesos/usuarios.
  - `--ipc=host` + `--shm-size` para `DataLoader` con workers.
  - Modo `detached` + `tmux` dentro del contenedor para corridas largas que sobrevivan
    cortes de SSH.
- **Pendiente de decidir en este spec** (no asumir): dónde viven los datasets pesados
  (verificar permisos, no asumir por espacio libre), convención de puertos por proyecto,
  ubicación de la(s) imagen(es) base compartida(s), gestión de la reserva de GPU entre
  proyectos/usuarios concurrentes, y el enfoque para proyectos sin GPU (web, mobile).

**Principio rector:** cada proyecto es responsable de su propio `docker/Dockerfile` y
`docker/run.sh` dentro de su repo (como ya hace `fno_co2`); este spec define únicamente lo
que se instala **una sola vez a nivel de sistema** y las convenciones **compartidas** entre
proyectos para que no colisionen entre sí (puertos, nombres de contenedor, GPU, disco).

---

## Fase 0 — Verificación del estado base del sistema

1. Confirmar versión de Ubuntu (`lsb_release -a`), driver NVIDIA (`nvidia-smi`) y que
   `docker --version` / `nvidia-ctk --version` reflejan lo ya instalado (parte de esto
   puede existir ya de la sesión de `fno_co2`; no reinstalar sin verificar primero).
2. Confirmar qué otros usuarios/proyectos corren activamente en la máquina
   (`docker ps -a` de todos, no solo del usuario propio, si los permisos lo permiten;
   `who`, `nvidia-smi` para procesos de GPU de otros usuarios).
3. Documentar en `docs/servidor.md` (Fase 7) el inventario inicial: GPUs físicas
   disponibles (índices), espacio en disco por partición relevante (`df -h`), y usuarios
   activos conocidos.

**Verificación:** `docs/servidor.md` tiene una sección "Estado inicial" con esta
información, fechada.

---

## Fase 1 — Instalación base a nivel de sistema (una sola vez)

**Dónde:** la workstation misma, fuera de cualquier repo de proyecto.

1. Docker Engine (`docker-ce`, `docker-ce-cli`, `containerd.io`,
   `docker-compose-plugin`) vía el repositorio oficial de Docker — **no** el paquete
   `docker.io` de Ubuntu (versión desactualizada).
2. NVIDIA Container Toolkit (`nvidia-ctk runtime configure --runtime=docker` +
   restart de `docker`).
3. Usuario del sistema agregado al grupo `docker` (`usermod -aG docker $USER`).
   **⚠️ Nota de seguridad:** pertenecer al grupo `docker` equivale en la práctica a acceso
   `root` sin sudo (se puede montar `/` del host dentro de un contenedor). Aceptable aquí
   porque el usuario ya tiene `sudo`, pero **no replicar** este paso para cuentas de otros
   colaboradores sin que sean conscientes de esa equivalencia.
4. Verificación: `docker run hello-world` y
   `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`.

**Verificación:** ambos comandos de (4) corren sin error; `docker info` no reporta el
runtime NVIDIA como faltante.

---

## Fase 2 — Verificación y ubicación del almacenamiento de datos pesados

**Dónde:** la workstation; todos los discos detectados en el inventario (punto 1), no
solo el primero que aparece con espacio libre.

1. **Inventario completo de discos**, montados o no — un disco con espacio libre visto
   solo en `df -h` puede no ser el único candidato, y puede haber discos sin montar que
   ya tienen dueño:
   ```bash
   lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINTS
   df -hT
   ```
   Hallazgo real en esta workstation (2026-07-04): 4 discos de ~10.9T (`sda`–`sdd`), pero
   solo 2 montados — `sda2` → `/media/imagenesmedicas/DATA1`, `sdb2` → `/media/nlp/DATA2`.
   `sdc2` y `sdd2` **no** aparecían montados.
2. **Antes de asumir que un disco sin montar está libre**, revisar si ya tiene filesystem
   y etiqueta (puede estar reservado aunque no esté en uso activo):
   ```bash
   sudo blkid /dev/sdc2 /dev/sdd2   # (ajustar a las particiones sin montar del inventario)
   ```
   Hallazgo real: `sdc2` y `sdd2` ya estaban formateados (`TYPE="ntfs"`) con las etiquetas
   `DATA3` y `DATA4` — mismo patrón de nombres que `DATA1`/`DATA2`. Esto indica que **no son
   discos libres sin dueño**: siguen la misma convención de asignación por grupo/proyecto,
   probablemente ya reservados para otros dos grupos aunque no estén montados hoy.
3. **Verificar contenido sin alterar el disco** (montaje de solo lectura, temporal, sin
   tocar `/etc/fstab`) — el listado de primer nivel no basta; confirmar tamaño real usado
   y revisar recursivamente cualquier carpeta de papelera/residuo antes de concluir que
   está vacío:
   ```bash
   sudo mkdir -p /mnt/check_<label>
   sudo mount -o ro /dev/<particion> /mnt/check_<label>
   ls -la /mnt/check_<label>
   sudo du -sh /mnt/check_<label>
   sudo find /mnt/check_<label> -type f 2>/dev/null
   sudo umount /mnt/check_<label>
   ```
   Hallazgo real: `sdc2` (`DATA3`) montó sin error, con `du -sh` = 4.0K y `find` sin
   archivos dentro de `.Trash-1003` → confirmado vacío. `sdd2` (`DATA4`) **no montó**:
   `$MFTMirr does not match $MFT` (NTFS inconsistente / posible falla de hardware) — no
   se forzó el montaje ni se intentó reparar (`ntfsfix`/`chkdsk`) sin autorización;
   queda como pendiente a reportar al administrador (ver `docs/servidor.md`).
4. Un disco con mucho espacio libre y ya montado (p. ej. `/media/nlp/DATA2`) **tampoco es
   automáticamente utilizable** — verificar dueño/grupo/permisos y confirmar con una prueba
   de escritura real, no inferir acceso solo de `df -h`:
   ```bash
   ls -la /media/nlp/DATA2
   id <usuario>
   touch /media/nlp/DATA2/.write_test_<usuario> && echo "OK: puedo escribir" && rm /media/nlp/DATA2/.write_test_<usuario>
   ```
5. **Ningún disco con patrón `DATA<N>` se reclama para un proyecto nuevo sin confirmar
   con quien administre la workstation** — la convención de nombres (`imagenesmedicas`,
   `nlp`, y los ya preparados `DATA3`/`DATA4`) indica asignación por grupo, no espacio de
   uso libre general.
6. Si un disco se libera y se confirma su uso, evaluar reformatear de NTFS a ext4 (mejor
   rendimiento y permisos POSIX nativos en Linux) — **solo tras confirmar que está vacío y
   disponible**, nunca antes.

   **Estado real — `DATA3` (`/dev/sdc2`):** autorizado por el administrador del lab;
   verificado vacío en el punto 3 (`du -sh` = 4.0K, `find` sin archivos en
   `.Trash-1003`). Punto de montaje elegido: `/media/sosagro4c/DATA3` (por usuario, no
   por proyecto, para reutilizarlo en otros trabajos futuros).

   **⚠️ Incidente durante la primera ejecución:** el paso de `mkfs.ext4` (el marcado como
   destructivo/irreversible) se saltó por error — se pasó directo de desmontar a crear el
   punto de montaje. El disco montó igual por autodetección de NTFS (sin dar error), y se
   agregó una línea a `/etc/fstab` declarando `ext4` sobre un disco que en realidad seguía
   siendo `ntfs` — un desajuste que podía colgar el arranque en el próximo reboot. No hubo
   pérdida de datos (el disco seguía vacío), pero fue necesario corregir: desmontar, borrar
   la línea de `fstab` incorrecta, y recién ahí correr el `mkfs.ext4` real. **Lección para
   cualquier disco futuro:** verificar con `blkid` que el `TYPE` ya sea el filesystem
   destino *después* de correr `mkfs`, antes de tocar `fstab`.

   **Comandos exactos de corrección — ya ejecutados, en este orden:**
   ```bash
   # 1. Desmontar
   sudo umount /media/sosagro4c/DATA3
   ```
   ```bash
   # 2. Sacar la linea incorrecta agregada antes (ext4 declarado sobre ntfs real)
   sudo sed -i '/\/media\/sosagro4c\/DATA3/d' /etc/fstab
   ```
   ```bash
   # 3. ⚠️ IMPORTANTE — DESTRUCTIVO E IRREVERSIBLE (el paso que se habia saltado).
   #    Corrido solo porque ya se cumplian ambas condiciones:
   #      a) confirmado vacio (punto 3: du -sh = 4.0K, find sin archivos)
   #      b) autorizado explicitamente por el administrador del lab
   sudo mkfs.ext4 -L DATA3 /dev/sdc2
   ```
   ```bash
   # 4. Montar y dar el usuario propio como dueno
   sudo mount /dev/sdc2 /media/sosagro4c/DATA3
   sudo chown sosagro4c:sosagro4c /media/sosagro4c/DATA3
   ```
   ```bash
   # 5. UUID real, ya como ext4 (cambio respecto al UUID de cuando era ntfs)
   sudo blkid /dev/sdc2
   # -> UUID="160da076-4ee3-48e3-a477-7c8d9e8b3847"
   ```
   ```bash
   # 6. Linea correcta en /etc/fstab -- con "nofail" (por el incidente de arriba: si
   #    este disco vuelve a fallar, el arranque del sistema no debe quedar colgado
   #    esperandolo)
   echo 'UUID=160da076-4ee3-48e3-a477-7c8d9e8b3847  /media/sosagro4c/DATA3  ext4  defaults,nofail  0  2' | sudo tee -a /etc/fstab
   sudo systemctl daemon-reload   # systemd avisa que hay que resincronizar tras editar fstab
   ```
   ```bash
   # 7. Verificar que el fstab quedo bien SIN necesidad de reiniciar
   sudo mount -a && df -hT /media/sosagro4c/DATA3
   # -> /dev/sdc2  ext4  11T  28K  11T  1%  /media/sosagro4c/DATA3
   ```

   **Resultado confirmado (2026-07-04):** `DATA3` reformateado a ext4, montado en
   `/media/sosagro4c/DATA3`, dueño `sosagro4c:sosagro4c`, persistente vía `/etc/fstab`
   con `nofail`. Disponible para datos de cualquier proyecto propio.
7. Decisión de ubicación de datos **por proyecto**: cada proyecto define en su propio
   `docker/run.sh` (o script equivalente) dónde monta `data/raw`/`data/processed` — dentro
   del propio repo (`~/proyectos/<nombre>/data/`, patrón actual de `fno_co2`) o en el disco
   compartido asignado, vía un volumen adicional apuntando al subdirectorio propio. Ambas
   son válidas; la elección depende del tamaño real del dataset, del espacio libre en
   `/home` y de qué disco quede efectivamente disponible — no es una regla única para
   todos los proyectos.
8. Registrar en `docs/servidor.md` (tabla de discos de la Fase 0) el inventario completo
   (incluidos los discos sin montar y su estado), qué proyecto usa qué ubicación, y los
   permisos verificados.

**Por qué es su propia fase y no un paso de la Fase 0:** la Fase 0 solo *observa* el estado
del sistema; decidir dónde escriben los proyectos y verificar que se puede hacer con
seguridad en una máquina multiusuario con discos ya asignados por grupo es una decisión
con consecuencias (puede pisar datos o reservas de otro grupo), así que se separa
explícitamente.

**Verificación:** inventario completo de discos (montados y sin montar) documentado;
`blkid` + montaje read-only revisados para cualquier disco sin montar antes de
considerarlo disponible; `ls -la` + prueba de escritura confirmados sin error para el
disco finalmente elegido; `docs/servidor.md` registra explícitamente dónde vive el dato
de cada proyecto y por qué, antes de que ese proyecto escriba ahí datos reales.

---

## Fase 3 — Estructura de directorios del usuario en la workstation

**Dónde:** `$HOME` del usuario en la workstation (no en este repo local de macOS).

1. Un único directorio raíz de trabajo, p. ej. `~/proyectos/`, con **un subdirectorio por
   proyecto** = un `git clone` independiente. El nombre del subdirectorio es el que trae
   el `git clone` por defecto (nombre del repo en GitHub) — no se fuerza un rename para
   que coincida con el nombre local del repo en otra máquina (p. ej. en la workstation el
   clone de `fno_co2` quedó como `~/proyectos/proyecto-simulaciones-inyeccion-co2/`, no
   `01-modelo-itm/`, y eso es aceptable):
   ```
   ~/proyectos/
   ├── proyecto-simulaciones-inyeccion-co2/   (clone de fno_co2, con su propio docker/)
   ├── <otro-proyecto-ml>/
   ├── <proyecto-web>/
   └── <proyecto-mobile>/
   ```
   Se descarta la estructura `~/workspace/{code,data,models,notebooks,outputs}` de la guía
   genérica inicial: para proyectos IA/ML ya se decidió (ver `01-Modelo-ITM/docker/`) que
   `data/`, `outputs/`, `docs/` viven **dentro** del propio repo (aunque gitignoreados), así
   que separarlos en carpetas hermanas solo introduce una segunda fuente de verdad.
2. Un directorio separado, fuera de cualquier repo, para artefactos verdaderamente
   compartidos entre proyectos si llegaran a existir (p. ej. un dataset grande reusado por
   dos proyectos distintos): `~/compartido/<nombre-dataset>/`, montado explícitamente como
   volumen adicional solo en los proyectos que lo necesiten. **No crear esta carpeta por
   adelantado** — solo cuando un caso real de reuso aparezca (evita acumular datos huérfanos).
3. Un directorio para las imágenes base compartidas (Fase 4):
   `~/proyectos/_docker-base/`.

**Verificación:** cada proyecto nuevo se clona bajo `~/proyectos/<nombre>/` y trae su
propio `docker/Dockerfile`; no hay carpetas `data/`/`models/` sueltas a nivel de `$HOME`
sin un proyecto que las reclame.

---

## Fase 4 — Imágenes base compartidas por tipo de proyecto

**Dónde:** `~/proyectos/_docker-base/` en la workstation — **carpeta suelta, sin
versionar en git** (decisión tomada: es un solo `Dockerfile` que cambia poco; se puede
crear un repo dedicado más adelante si hace falta historial).

**Alcance decidido:** solo `base-cuda:py312` por ahora — es lo único que `fno_co2`
necesita hoy. `base-node` (proyectos web) y la decisión de mobile quedan explícitamente
pendientes hasta que exista un proyecto real que las necesite (mismo criterio de "no
crear por adelantado" ya usado en la Fase 2 para `~/compartido/`).

**Decisión de diseño (ajusta el texto original de este spec):** la creación del usuario
no-root **no** vive en `base-cuda` — se mantiene en el `Dockerfile` de cada proyecto
(como ya hace `fno_co2`). Motivo: el daemon de Docker es compartido entre varios usuarios
de la workstation; si el UID/GID quedara fijado al construir la imagen base, otro usuario
que la reutilice heredaría un UID que no es el suyo. `base-cuda` se mantiene
deliberadamente genérica — CUDA/cuDNN, Python 3.12 y herramientas de sistema — y cada
proyecto agrega su propio usuario sobre esa base, igual que hace hoy.

### Comandos exactos — construir `base-cuda:py312` en la workstation

```bash
# 1. Crear la carpeta (sin versionar en git, segun lo decidido)
mkdir -p ~/proyectos/_docker-base/base-cuda
cd ~/proyectos/_docker-base/base-cuda
```

```bash
# 2. Escribir el Dockerfile de la base compartida
cat > Dockerfile << 'EOF'
# Imagen base compartida GPU (IA/ML/DL/LLM/NLP) -- spec-005 Fase 4.
# Generica a proposito: SIN usuario no-root (eso lo agrega cada proyecto en su propio
# Dockerfile, ver docker/Dockerfile de fno_co2) y SIN dependencias de un proyecto
# especifico (sin torch/transformers/etc.).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        git wget curl vim htop tmux ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /workspace
CMD ["bash"]
EOF
```

```bash
# 3. Construir la imagen (una sola vez; se reconstruye solo si este Dockerfile cambia)
docker build -t base-cuda:py312 ~/proyectos/_docker-base/base-cuda
```

```bash
# 4. Verificar: CUDA, Python 3.12 y herramientas presentes, sin torch (a proposito)
docker run --rm --gpus all base-cuda:py312 bash -c "nvidia-smi && python3 --version && git --version && tmux -V"
```

**Ejecutado en la workstation (2026-07-04) — ajustes respecto al texto original:**

- El tag `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04` (versión original de este spec)
  **no existe** en Docker Hub — CUDA 12.4.1 solo publica variantes `ubuntu22.04`/`ubuntu20.04`.
  Decisión del usuario: mantener `ubuntu24.04` y subir a la versión más cercana que sí la
  tiene, `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` (el driver de la workstation,
  580.159.03, soporta hasta CUDA 13.0, así que no hay problema de compatibilidad).
- Ubuntu 24.04 aplica PEP 668 (`externally-managed-environment`): `pip install` a nivel de
  sistema falla sin más por defecto — algo que no ocurría con la imagen `pytorch/pytorch`
  usada previamente. Fix aplicado en `base-cuda/Dockerfile`: `ENV PIP_BREAK_SYSTEM_PACKAGES=1`
  (equivalente a `pip install --break-system-packages`, aceptable porque el contenedor es
  un entorno descartable dedicado a un solo propósito, no un sistema compartido).
- Verificado: `base-cuda:py312` construida (CUDA 12.6.3, Python 3.12.3, git 2.43.0, tmux 3.4,
  sin `torch`). `fno-co2:dev` reconstruida sobre esta base — `docker history` confirma las
  capas heredadas de `nvidia/cuda:12.6.3-...-ubuntu24.04`, y
  `torch.cuda.is_available()` → `True` (`torch 2.12.1+cu130`) dentro del contenedor del
  proyecto.

### Comandos exactos — migrar `fno_co2` para partir de la base compartida

El `docker/Dockerfile` de `fno_co2` ya fue actualizado (ver commit en `development`) para
usar `FROM base-cuda:py312` en vez de `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`
directo. `torch` se sigue instalando igual que antes, vía `pip install -e ".[dev]"`
(ya está declarado en `pyproject.toml`), así que no hace falta ningún cambio manual de
dependencias — solo reconstruir:

```bash
cd ~/proyectos/proyecto-simulaciones-inyeccion-co2
git pull origin development
./docker/run.sh build
```

```bash
# Verificar que el contenedor del proyecto sigue viendo la GPU y ahora usa la base
# compartida (docker history muestra las capas heredadas de base-cuda:py312)
docker history fno-co2:dev | grep -i "base-cuda\|FROM" 
docker run --rm --gpus all fno-co2:dev python -c "import torch; print(torch.cuda.is_available())"
```

**Por qué una base compartida y no una imagen por proyecto desde cero:** evita reconstruir
CUDA/cuDNN en cada proyecto (tiempo de build) y centraliza el pin de versión de
CUDA/driver en un solo lugar en vez de N `Dockerfile`s divergentes. `base-node` (web) y
mobile quedan fuera de esta implementación hasta que haya un proyecto real que los
necesite — ver nota de alcance arriba.

**Verificación:** `docker images` muestra `base-cuda:py312` construida una sola vez; el
`Dockerfile` de `fno_co2` arranca con `FROM base-cuda:py312` y no reinstala CUDA/Python;
`torch.cuda.is_available()` sigue devolviendo `True` dentro del contenedor del proyecto.

---

## Fase 5 — Convención de puertos y nombres de contenedor [DONE]

**Dónde:** `docs/servidor.md` (Fase 7).

**Nota de ubicación (ajusta el texto original):** la Fase 3 descartó replicar la
estructura `09-Proyecto-Deep-Learning/` en la workstation (solo existe
`~/proyectos/<proyecto>/` y `~/proyectos/_docker-base/`), así que no hay una carpeta de
"raíz del ecosistema" literal donde colgar `docs/servidor.md`. Decisión del usuario
(2026-07-04): vive en `~/proyectos/_docs/servidor.md` — carpeta hermana a
`_docker-base/`, mismo criterio (suelta, sin versionar en git, documenta la máquina no
un proyecto). El archivo ya se creó con esta sección; las demás secciones de la Fase 7
quedan pendientes y se agregan incrementalmente al mismo archivo.

1. Con múltiples proyectos corriendo contenedores simultáneamente en la misma máquina,
   puertos como `8888` (Jupyter) o `6006` (TensorBoard) van a colisionar si dos proyectos
   los exponen a la vez. Reservar un rango de puertos por proyecto y documentarlo, p. ej.:
   `proyecto-simulaciones-inyeccion-co2` (`fno_co2`) → `8888`/`6006`, siguiente proyecto
   → `8889`/`6007`, etc.
2. Nombres de contenedor prefijados por proyecto (`fno-<sesion>` ya es el patrón en
   `fno_co2`; generalizar a `<slug-proyecto>-<sesion>`) para que `docker ps` sea legible
   con varios proyectos activos a la vez.

**Ejecutado (2026-07-04):** `~/proyectos/_docs/servidor.md` creado con la tabla
`proyecto | slug | prefijo contenedor | puerto Jupyter | puerto TensorBoard | notas`.
`fno_co2` registrado con prefijo `fno-<sesion>` (ya implementado en `docker/run.sh`,
sin cambios de código necesarios) y puertos `8888`/`6006` reservados preventivamente
(el proyecto no expone Jupyter/TensorBoard hoy). Próximo proyecto → `8889`/`6007`.

**Verificación:** `docs/servidor.md` tiene una tabla `proyecto | puertos reservados |
prefijo de contenedor`, actualizada cada vez que se agrega un proyecto nuevo. —
Cumplido: tabla creada en `~/proyectos/_docs/servidor.md`.

---

## Fase 6 — Gestión de GPU compartida entre proyectos y usuarios [DONE]

1. Antes de lanzar cualquier contenedor con `--gpus device=<N>`, correr `nvidia-smi` y
   revisar procesos activos de otros usuarios en esa GPU.
2. Si la máquina tiene una sola GPU física y varios proyectos/usuarios la disputan,
   documentar en `docs/servidor.md` una tabla simple de reserva manual (`GPU 0 | proyecto |
   usuario | desde–hasta`) — no hay guarda automática de scheduling; es disciplina de
   proceso, igual que la del split train/test en `spec-001` de `fno_co2`.
3. Evaluar (no implementar todavía, **requiere decisión del usuario** si se vuelve
   necesario) MPS o time-slicing de NVIDIA solo si la contención real por GPU aparece en
   la práctica — no instalar preventivamente sin un problema concreto que lo justifique.

**Ejecutado (2026-07-04):** inventario confirmado — 1 sola GPU física (`GPU 0: NVIDIA
RTX 6000 Ada Generation`, 49140 MiB), sin procesos de cómputo activos al momento de
verificar. Otros usuarios activos en la máquina (`who`): `imagenesmedicas`, `nlp`, además
de `sosagro4c` — mismo patrón de asignación por grupo que los discos `DATA1`–`DATA4`
(Fase 2). Tabla de reserva manual creada en `~/proyectos/_docs/servidor.md`
(sección "Reserva de GPU compartida"), vacía por ahora (sin corridas largas activas).
MPS/time-slicing: evaluado, no instalado — no hay contención real hoy que lo justifique.

**Verificación:** no automatizable — revisión de proceso; se aplica documentando cada
corrida larga en `docs/servidor.md`. — Tabla y proceso documentados en
`~/proyectos/_docs/servidor.md`.

---

## Fase 7 — Persistencia de procesos y documentación del servidor [DONE]

1. Convención ya validada en `fno_co2`: corridas largas en modo `docker run -d` +
   `tmux` dentro del contenedor, reconexión vía `docker exec -it <nombre> tmux attach`.
   Se generaliza a todos los proyectos con contenedores de larga duración (entrenamientos,
   servidores de desarrollo web persistentes).
2. Crear `docs/servidor.md` en la raíz del ecosistema (`09-Proyecto-Deep-Learning/docs/`,
   fuera de cualquier repo de proyecto individual, ya que documenta la máquina, no un
   proyecto) con las secciones: estado inicial (Fase 0), ubicación de almacenamiento
   (Fase 2), imágenes base (Fase 4), tabla de puertos/contenedores (Fase 5), tabla de
   reserva de GPU (Fase 6), y changelog de cambios al entorno base (cuándo se actualizó
   `base-cuda`, por qué).

**Nota de ubicación (ver también Fase 5):** la Fase 3 descartó replicar
`09-Proyecto-Deep-Learning/` en la workstation, así que el archivo vive en
`~/proyectos/_docs/servidor.md` (decisión del usuario, misma carpeta ya usada para la
sección de la Fase 5).

**Ejecutado (2026-07-04):** `~/proyectos/_docs/servidor.md` completado con las 5
secciones — estado inicial, almacenamiento, imágenes base, puertos/contenedores, reserva
de GPU — más un changelog. Contenido verificado contra el estado real del sistema en el
momento de escribirlo (no copiado a ciegas del spec): `lsb_release`, `docker --version`,
`nvidia-smi`, `nvidia-ctk --version`, `lsblk`/`df -h` de todos los discos, y
`docker images` de `base-cuda:py312`/`fno-co2:dev`.

**Verificación:** `docs/servidor.md` existe y tiene las 5 secciones; se actualiza (no se
sobrescribe) cada vez que se agrega un proyecto o cambia la imagen base. — Cumplido en
`~/proyectos/_docs/servidor.md`.

---

## 1. Archivos / directorios impactados (resumen)

| Ruta | Fase | Naturaleza |
|---|---|---|
| Workstation: paquetes `docker-ce`, `nvidia-container-toolkit` | 1 | Instalación de sistema, **⚠️ requiere confirmación explícita** antes de instalar/actualizar |
| Workstation: `/media/sosagro4c/DATA3` (`sdc2`) | 2 | **Resuelto** — reformateado NTFS→ext4, montado, dueño `sosagro4c`, persistente en `fstab` con `nofail` |
| Workstation: `sdd2` (`DATA4`, NTFS corrupto) | 2 | **Pendiente** — reportar al administrador, no reparar sin autorización |
| Workstation: `~/proyectos/` | 3 | Convención de directorios — reemplaza la estructura `~/workspace/*` genérica |
| Workstation: `~/proyectos/_docker-base/base-cuda/` | 4 | Nuevo — `Dockerfile` de `base-cuda:py312`, carpeta suelta sin versionar en git |
| `01-Modelo-ITM/docker/Dockerfile` | 4 | Actualizado — ahora parte de `FROM base-cuda:py312` en vez de `pytorch/pytorch:...` directo |
| `09-Proyecto-Deep-Learning/docs/servidor.md` | 0, 2, 5, 6, 7 | Nuevo — registro del entorno a nivel ecosistema |

---

## 2. Riesgos y precondiciones

- **Grupo `docker` = acceso root-equivalente:** aceptable para el usuario principal (ya
  tiene `sudo`), pero si en el futuro se crean cuentas para colaboradores en la misma
  máquina, agregarlos al grupo `docker` sin más consideración les da control total del
  host — evaluarlo explícitamente en ese momento, no asumir que es automático.
- **GPU física única compartida:** sin scheduling automático, dos corridas simultáneas de
  proyectos distintos pueden competir por VRAM y degradar ambas silenciosamente (no hay
  error explícito, solo lentitud/OOM). La Fase 6 documenta el proceso manual; si se vuelve
  un problema recurrente, esto necesitaría revisarse (MPS, colas de jobs) — **fuera de
  alcance de este spec**.
- **Disco compartido para datasets:** el prefijo `/media/<grupo>/` indica asignación por
  grupo/usuario — **no escribir sin verificar permisos primero** (Fase 2). Espacio libre
  no implica acceso de escritura ni que sea territorio disponible. Resuelto para `DATA3`
  (autorizado, verificado vacío, reformateado); `DATA4` sigue pendiente por filesystem
  NTFS corrupto — no intentar reparar (`ntfsfix`/`chkdsk`) sin autorización explícita del
  administrador, riesgo de agravar una falla de hardware real o perder datos ajenos
  recuperables.
- **Ubicación de `_docker-base/`:** **resuelto** — carpeta suelta sin versionar en git
  (decisión del usuario). Si en el futuro hace falta historial de cambios de la imagen
  base, se puede migrar a un repo dedicado sin romper nada (el `Dockerfile` no depende de
  estar versionado para funcionar).
  <a id="mobile-decision"></a>
- **Alcance mobile no resuelto:** este spec no asume que la workstation Ubuntu sirve para
  compilar iOS — **requiere decisión explícita del usuario** antes de crear una imagen
  base para mobile (`base-node`/mobile quedaron fuera de esta implementación de la Fase 4;
  ver Fase 4).
- **Todo lo de Fase 1 (instalación de paquetes de sistema) requiere confirmación explícita
  antes de ejecutarse**, igual que cualquier instalación de dependencias según las reglas
  ya establecidas por proyecto (ver `CLAUDE.md` de `01-Modelo-ITM` §Dependencias, que aplica
  el mismo criterio aunque este spec sea de alcance más amplio).

---

## 3. Criterios de aceptación

- [x] Docker Engine + NVIDIA Container Toolkit instalados y verificados (Fase 1) — ya
      presentes en la workstation, GPU passthrough confirmado con `docker run --gpus all`.
- [x] Permisos del disco elegido para datos pesados verificados con prueba de escritura
      real (no solo `df -h`); ubicación de datos por proyecto documentada en
      `docs/servidor.md` (Fase 2). — `DATA3` autorizado, verificado vacío, reformateado
      a ext4 y montado en `/media/sosagro4c/DATA3` (persistente en `fstab`). `DATA4`
      sigue pendiente, reportado al administrador.
- [x] `~/proyectos/` existe con la convención un-subdirectorio-por-proyecto (Fase 3) —
      `fno_co2` clonado en `~/proyectos/proyecto-simulaciones-inyeccion-co2/`.
- [x] Al menos la imagen base `base-cuda:py312` construida y utilizable por
      `FROM` desde un `Dockerfile` de proyecto (Fase 4) — construida y verificada en la
      workstation (2026-07-04); `fno-co2:dev` reconstruida sobre ella con
      `torch.cuda.is_available() == True`. Ver ajuste de tag CUDA y fix de PEP 668 en la
      Fase 4.
- [x] Decisión explícita del usuario registrada sobre si `_docker-base/` se versiona en
      git (Riesgos) — resuelto: carpeta suelta, sin git. Alcance mobile sigue pendiente
      (no bloquea `base-cuda`).
- [x] `docs/servidor.md` existe en la raíz del ecosistema con las 5 secciones de la
      Fase 7 — en `~/proyectos/_docs/servidor.md` (ver nota de ubicación en Fase 5/7),
      completado y verificado contra el sistema real (2026-07-04).
- [x] Convención de puertos/nombres de contenedor documentada antes de correr un segundo
      proyecto con Docker en la misma máquina (Fase 5) — tabla creada en
      `~/proyectos/_docs/servidor.md` (2026-07-04).
