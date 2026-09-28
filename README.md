# Agente bilingüe / Bilingual agent

Un agente de IA de propósito general para la línea de comandos. Le pides una
tarea —en español o en inglés— y la completa: lee y escribe archivos, ejecuta
comandos, busca en la web y te cuenta qué ha hecho.

*A general-purpose AI agent for the command line. Ask it for a task — in
Spanish or English — and it gets it done: reads and writes files, runs
commands, searches the web, and reports back.*

---

## Español

### Instalación

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
```

### Uso

```bash
# Sesión interactiva
agent                 # o: python -m agent

# Una tarea y salir
agent "resume los cambios del último commit"

# Desde una tubería
echo "¿qué hace este script?" | agent -C ./scripts
```

En la sesión interactiva:

| Comando | Qué hace |
| --- | --- |
| `/help` | muestra la ayuda |
| `/clear` | olvida la conversación actual |
| `/lang en\|es` | cambia el idioma de la interfaz |
| `/tools` | lista las herramientas disponibles |
| `/cost` | tokens usados y coste estimado |
| `/exit` | salir |

Los comandos también responden a sus nombres en español: `/ayuda`, `/limpiar`,
`/idioma`, `/herramientas`, `/coste`, `/salir`.

### Sobre el bilingüismo

El agente **responde siempre en el idioma en que le escribes**, y cambia de
idioma a mitad de conversación si tú lo haces, sin anunciarlo. No traduce: el
español y el inglés son igual de nativos para él.

Además:

- Se adapta a tu variedad y registro (tú / vos / usted, vocabulario de tu país).
- Mantiene en inglés los tecnicismos que de verdad se usan en inglés —*commit*,
  *deploy*, *branch*, *pull request*— en lugar de forzar traducciones que nadie
  dice.
- Si mezclas idiomas, te los mezcla de vuelta al mismo nivel.
- Si le pides explícitamente un idioma, lo mantiene hasta que digas otra cosa.

`/lang` **no** afecta a esto: solo cambia el menú y los mensajes de la
interfaz. El idioma de las respuestas lo decides tú al escribir.

### Opciones

| Opción | Efecto |
| --- | --- |
| `-C, --directory RUTA` | directorio de trabajo (por defecto, el actual) |
| `--model ID` | modelo a usar (por defecto `claude-opus-5`) |
| `--effort {low,medium,high,xhigh,max}` | profundidad de razonamiento (por defecto `high`) |
| `--lang {en,es}` | idioma de la interfaz (por defecto, tu `LANG`) |
| `-y, --yes` | ejecuta comandos sin pedir confirmación |
| `--no-shell` | desactiva la ejecución de comandos |
| `--no-web` | desactiva la búsqueda web |
| `--no-thinking` | oculta el resumen del razonamiento |
| `--allow-outside` | permite tocar archivos fuera del directorio de trabajo |
| `--max-steps N` | tope de pasos por turno (por defecto 60) |

También se pueden fijar con variables de entorno: `AGENT_MODEL`, `AGENT_LANG`,
`AGENT_EFFORT`.

### Seguridad

- **Los archivos están acotados** al directorio de trabajo. Cualquier ruta que
  intente salir (`../`, rutas absolutas) se rechaza, salvo con `--allow-outside`.
- **Cada comando se confirma.** Responde `y` / `sí`, `n` / `no`, o `a` para no
  volver a preguntar en esa sesión. Si no hay terminal (tubería, CI), los
  comandos se deniegan salvo que pases `--yes`.
- Un comando rechazado no es un error: el agente lo sabe y busca otra vía.

---

## English

### Install

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
```

### Use

```bash
agent                                        # interactive session
agent "summarise the last commit"            # one task, then exit
echo "what does this script do?" | agent -C ./scripts
```

Slash commands: `/help`, `/clear`, `/lang en|es`, `/tools`, `/cost`, `/exit`.

### About the bilingual behaviour

The agent **always replies in the language you write in**, and follows you
silently if you switch mid-conversation. It does not translate — English and
Spanish are equally native to it. It matches your regional variety and
register, keeps the technical vocabulary practitioners actually use, and mirrors
code-switching back at the same level.

`/lang` only changes the menus and prompts of the CLI itself; the language of
the answers is decided by what you type.

### Options

See the Spanish table above — the flags are the same: `-C/--directory`,
`--model`, `--effort`, `--lang`, `-y/--yes`, `--no-shell`, `--no-web`,
`--no-thinking`, `--allow-outside`, `--max-steps`. Environment equivalents:
`AGENT_MODEL`, `AGENT_LANG`, `AGENT_EFFORT`.

### Safety

File access is confined to the working directory unless `--allow-outside` is
passed, and every shell command needs your approval (`y`/`n`/`a`, or `--yes` to
skip asking). Without a terminal, commands are denied rather than run blind.

---

## Cómo está hecho / How it works

```
agent/
  cli.py        REPL, argumentos, renderizado, aprobaciones
  core.py       el bucle: petición → herramientas → petición
  prompts.py    el system prompt (incluye las reglas de idioma)
  i18n.py       textos de la interfaz en es/en
  config.py     configuración y tarifas
  tools/
    base.py     sandbox de rutas, validación de entradas, ejecución
    files.py    read_file, write_file, edit_file, list_directory
    shell.py    run_command (con confirmación)
    web.py      web_search y web_fetch (se ejecutan en el servidor)
```

Detalles que importan:

- **Streaming siempre.** Ves el texto según se genera y no hay riesgo de
  *timeout* en respuestas largas.
- **Razonamiento adaptativo** (`thinking: adaptive`) con el resumen visible, y
  `effort` configurable.
- **El prefijo del system prompt está cacheado**, así que los turnos siguientes
  de una conversación son bastante más baratos. Todo lo que varía por sesión
  (directorio, fecha) va en un bloque aparte, después del punto de caché.
- **Las entradas de las herramientas se validan** contra su esquema antes de
  ejecutarse. Hace falta porque el agente pide `eager_input_streaming`, y con
  eso la API puede entregar una entrada truncada sin lanzar ningún error.
- **Llamadas en paralelo**: si el modelo pide varias herramientas en un turno,
  todos los resultados vuelven en un solo mensaje, como exige la API.
- **Ctrl-C a mitad de turno no rompe la sesión**: las llamadas a herramientas
  que quedaron colgando se cierran con un resultado de error antes del
  siguiente turno.

### Tests

```bash
pip install -e ".[dev]"
python -m pytest
```

88 tests, sin llamadas a la API: el cliente está sustituido por un doble que
reproduce los eventos del SDK.
