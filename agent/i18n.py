"""Bilingual user interface strings (English / Spanish).

Only the CLI chrome lives here. Anything the *model* reads -- the system
prompt, tool descriptions, tool results and error strings returned to the
model -- stays in English on purpose: the model is equally fluent in both and
a single canonical wording keeps the prompt cache prefix stable.
"""

from __future__ import annotations

import os

SUPPORTED = ("en", "es")
FALLBACK = "en"

STRINGS: dict[str, dict[str, str]] = {
    "banner": {
        "en": "Agent ready. Ask for anything, in English or Spanish.",
        "es": "Agente listo. Pide lo que necesites, en español o en inglés.",
    },
    "banner_hint": {
        "en": "Type /help for commands, /exit to quit.",
        "es": "Escribe /help para ver los comandos, /exit para salir.",
    },
    "workspace": {
        "en": "Workspace: {path}",
        "es": "Directorio de trabajo: {path}",
    },
    "prompt": {"en": "you", "es": "tú"},
    "agent_label": {"en": "agent", "es": "agente"},
    "goodbye": {"en": "Bye.", "es": "Hasta luego."},
    "cleared": {
        "en": "Conversation cleared.",
        "es": "Conversación borrada.",
    },
    "help_title": {"en": "Commands", "es": "Comandos"},
    "help_help": {"en": "show this help", "es": "muestra esta ayuda"},
    "help_clear": {
        "en": "forget the conversation so far",
        "es": "olvida la conversación actual",
    },
    "help_lang": {
        "en": "switch the interface language (en/es)",
        "es": "cambia el idioma de la interfaz (en/es)",
    },
    "help_cost": {
        "en": "estimated token usage and cost",
        "es": "uso de tokens y coste estimado",
    },
    "help_tools": {
        "en": "list the tools the agent can use",
        "es": "lista las herramientas del agente",
    },
    "help_exit": {"en": "quit", "es": "salir"},
    "help_language_note": {
        "en": "The agent always replies in the language you write in; /lang only "
        "changes this menu and the prompts.",
        "es": "El agente siempre responde en el idioma en que escribes; /lang solo "
        "cambia este menú y los mensajes de la interfaz.",
    },
    "lang_set": {
        "en": "Interface language: English.",
        "es": "Idioma de la interfaz: español.",
    },
    "lang_unknown": {
        "en": "Unknown language '{lang}'. Available: {options}.",
        "es": "Idioma desconocido '{lang}'. Disponibles: {options}.",
    },
    "tools_title": {"en": "Available tools", "es": "Herramientas disponibles"},
    "thinking_label": {"en": "thinking", "es": "pensando"},
    "tool_running": {"en": "{name}({summary})", "es": "{name}({summary})"},
    "tool_ok": {"en": "done", "es": "hecho"},
    "tool_failed": {"en": "failed", "es": "error"},
    "confirm_command": {
        "en": "Run this command?",
        "es": "¿Ejecuto este comando?",
    },
    "confirm_choices": {
        "en": "[y] yes  [n] no  [a] yes, and don't ask again this session",
        "es": "[y] sí  [n] no  [a] sí, y no volver a preguntar en esta sesión",
    },
    "confirm_prompt": {"en": "choice", "es": "opción"},
    "denied": {
        "en": "Command not run (you declined).",
        "es": "Comando no ejecutado (lo has rechazado).",
    },
    "denied_noninteractive": {
        "en": "Command not run: no terminal to confirm on. Re-run with --yes to "
        "allow commands without asking.",
        "es": "Comando no ejecutado: no hay terminal para confirmar. Usa --yes para "
        "permitir comandos sin preguntar.",
    },
    "interrupted": {
        "en": "Interrupted. The partial answer above is kept in the conversation.",
        "es": "Interrumpido. La respuesta parcial de arriba se mantiene en la conversación.",
    },
    "max_steps": {
        "en": "Stopped after {n} steps without finishing. Ask me to continue if needed.",
        "es": "Me he detenido tras {n} pasos sin terminar. Pídeme que continúe si hace falta.",
    },
    "truncated": {
        "en": "(response hit the output limit and was cut off)",
        "es": "(la respuesta alcanzó el límite de salida y se cortó)",
    },
    "refusal": {
        "en": "The model declined to answer this request{detail}.",
        "es": "El modelo ha declinado responder a esta petición{detail}.",
    },
    "cost_title": {"en": "Session usage", "es": "Uso de la sesión"},
    "cost_input": {"en": "input tokens", "es": "tokens de entrada"},
    "cost_output": {"en": "output tokens", "es": "tokens de salida"},
    "cost_cache_read": {"en": "cached tokens read", "es": "tokens leídos de caché"},
    "cost_cache_write": {"en": "tokens written to cache", "es": "tokens escritos en caché"},
    "cost_total": {"en": "estimated cost", "es": "coste estimado"},
    "cost_unknown_model": {
        "en": "No price list for model '{model}'; showing token counts only.",
        "es": "No hay tarifas para el modelo '{model}'; solo se muestran los tokens.",
    },
    "no_api_key": {
        "en": "No Anthropic credentials found. Set ANTHROPIC_API_KEY, or run "
        "`ant auth login` if you have the Anthropic CLI.",
        "es": "No se han encontrado credenciales de Anthropic. Define ANTHROPIC_API_KEY, "
        "o ejecuta `ant auth login` si tienes el CLI de Anthropic.",
    },
    "api_error": {"en": "API error: {detail}", "es": "Error de la API: {detail}"},
    "rate_limited": {
        "en": "Rate limited. Try again in {seconds}s.",
        "es": "Límite de peticiones alcanzado. Inténtalo de nuevo en {seconds}s.",
    },
    "connection_error": {
        "en": "Could not reach the API. Check your network connection.",
        "es": "No se ha podido contactar con la API. Revisa tu conexión de red.",
    },
    "empty_response": {
        "en": "(the model returned no text)",
        "es": "(el modelo no ha devuelto texto)",
    },
}


def normalize(lang: str | None) -> str | None:
    """Map a locale-ish string onto a supported language code."""
    if not lang:
        return None
    code = lang.strip().lower().replace("_", "-").split(".")[0].split("-")[0]
    return code if code in SUPPORTED else None


def detect_language() -> str:
    """Pick a UI language from the environment, defaulting to English."""
    for var in ("AGENT_LANG", "LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        if code := normalize(os.environ.get(var)):
            return code
    return FALLBACK


class Translator:
    """Looks up UI strings for the active language."""

    def __init__(self, lang: str | None = None) -> None:
        self.lang = normalize(lang) or detect_language()

    def set_language(self, lang: str) -> bool:
        code = normalize(lang)
        if code is None:
            return False
        self.lang = code
        return True

    def __call__(self, key: str, **kwargs: object) -> str:
        entry = STRINGS.get(key)
        if entry is None:
            return key
        template = entry.get(self.lang) or entry[FALLBACK]
        return template.format(**kwargs) if kwargs else template
