# Local Note Illustrator

Aplicación de escritorio local en Python para **leer archivos `.docx` desde carpetas locales** y **generar imágenes con IA open-source (Diffusers + Stable Diffusion XL (SDXL))**, guardando los resultados en la **misma subcarpeta del documento origen**.

Está pensada para ejecutarse en **Windows desde Anaconda Prompt** con Python 3.10.

---

## ¿Qué hace la aplicación?

1. Actualiza el repositorio local con `git pull` desde la GUI.
2. Permite seleccionar carpeta raíz local.
3. Escanea archivos `.docx` (ignora temporales `~$`).
4. Lee texto con `python-docx`.
5. Construye prompts visuales localmente (sin APIs pagas ni LLM externos).
6. Genera 1 o 2 imágenes por documento con Diffusers usando SDXL.
7. Guarda imágenes en la misma carpeta del `.docx`.
8. Muestra logs, progreso y estado en la GUI.
9. Permite abrir carpeta seleccionada y carpeta de logs desde la GUI.
10. Permite exportar parámetros de ejecución en JSON desde la GUI.

---

## Estructura del repositorio

```text
local-note-illustrator/
  README.md
  .gitignore
  .env.example
  requirements.txt
  run_app.py
  app/
    __init__.py
    gui.py
    config.py
    logger.py
    repo_updater.py
    docx_reader.py
    prompt_builder.py
    image_generator.py
    scanner.py
    utils.py
  docs/
  logs/
```

---


## Modelo SDXL por defecto

- La app utiliza **stabilityai/stable-diffusion-xl-base-1.0** como modelo principal y único flujo de generación.
- En la primera ejecución se descarga el modelo (requiere internet y puede tardar).
- El modo CPU funciona, pero es significativamente más lento que CUDA.
- Podés forzar CPU con `FORCE_CPU=true` en `.env`.
- Desde la GUI podés usar **Exportar parámetros** para guardar `logs/run_config_YYYY-MM-DD_HH-MM-SS.json`.

## Requisitos

- Windows 10/11
- Anaconda o Miniconda
- Python 3.10
- Git instalado y repo clonado localmente
- GPU NVIDIA + CUDA (recomendado) para generación rápida
  - Si no hay CUDA, usa CPU (más lento)

---

## Instalación (Anaconda)

> Ejecutar en **Anaconda Prompt** dentro de la carpeta del proyecto.

```bash
conda create -n local-note-illustrator python=3.10 -y
conda activate local-note-illustrator
pip install -r requirements.txt
```

### Nota sobre PyTorch + CUDA
Si querés versión CUDA específica de PyTorch, podés reinstalar torch según tu entorno NVIDIA desde la documentación oficial de PyTorch.

---

## Configuración

1. Copiar archivo de ejemplo:

```bash
copy .env.example .env
```

2. Editar `.env` si querés cambiar:

- `MODEL_ID`
- `DEFAULT_NEGATIVE_PROMPT`
- `DEFAULT_NUM_IMAGES`
- `DEFAULT_STEPS`
- `DEFAULT_GUIDANCE_SCALE`
- `DEFAULT_WIDTH`
- `DEFAULT_HEIGHT`
- `OUTPUT_FORMAT`
- `LOG_DIR`
- `FORCE_CPU` (usar `true` para desactivar CUDA manualmente)

Si no existe `.env`, la app usa defaults internos sanos.

---

## Ejecución

```bash
python run_app.py
```

---

## Uso de la GUI

1. **Actualizar repo**: hace `git pull` de la rama activa.
2. **Seleccionar carpeta raíz**: elige la carpeta local con tus notas.
3. **Escanear documentos**: busca `.docx` (opción de subcarpetas).
4. **Generar imágenes**:
   - Lee cada `.docx`
   - Construye prompts positivos/negativos
   - Genera 1 o 2 imágenes por documento
5. **Abrir carpeta**: abre la carpeta raíz seleccionada.
6. **Abrir logs**: abre la carpeta de logs.

### Resultado esperado por documento
Si existe:

```text
D:\MisNotas\ProyectoA\nota_01.docx
```

la app puede generar:

```text
D:\MisNotas\ProyectoA\nota_01_img_01.png
D:\MisNotas\ProyectoA\nota_01_img_02.png
```

Siempre en la **misma subcarpeta** del `.docx` origen.

---

## Notas sobre GPU NVIDIA / CUDA

- La app valida CUDA de forma real antes de usarla (no solo `torch.cuda.is_available()`).
- Si detecta errores de compatibilidad (ej. `no kernel image is available for execution on the device`), hace fallback automático a CPU.
- Con CUDA usa `float16` para mejorar rendimiento/memoria.
- En CPU usa `float32`; es más lento pero funcional.
- Podés forzar CPU con `FORCE_CPU=true` en `.env`.
- La carga del pipeline es perezosa (lazy): solo se carga al generar.

---

## Troubleshooting

### 1) `git pull` falla por cambios locales
Guardá/cerrá cambios locales (`commit` o `stash`) y reintentá desde la GUI.

### 2) Error al cargar modelo Diffusers
- Revisar conexión a internet la primera vez (descarga modelo).
- Verificar espacio en disco y memoria VRAM/RAM.
- Probar menor resolución (ej. `768x768`) y menos `steps`.

### 3) `CUDA error: no kernel image is available for execution on the device`
Significa que la GPU no es compatible con la build CUDA/PyTorch instalada.
La app ahora detecta este caso y reintenta automáticamente en CPU para continuar el proceso.
Si querés evitar cualquier intento de CUDA, configurá `FORCE_CPU=true` en `.env`.

### 4) Generación muy lenta
- Confirmar instalación de PyTorch con CUDA para tu GPU.
- Reducir `DEFAULT_STEPS` y/o tamaño de imagen.

### 5) No encuentra `.docx`
- Verificar carpeta raíz correcta.
- Activar checkbox de subcarpetas.
- Revisar que no sean archivos temporales `~$...`.

---

## Licencia / Uso

Proyecto base para iteración local con GitHub/Codex. Ajustá el modelo y parámetros según tus recursos de hardware.
