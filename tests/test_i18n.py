"""Tests for the bilingual interface layer."""

from __future__ import annotations

import pytest

from agent.i18n import FALLBACK, STRINGS, SUPPORTED, Translator, detect_language, normalize


def test_every_string_exists_in_every_supported_language():
    missing = [
        (key, lang)
        for key, entry in STRINGS.items()
        for lang in SUPPORTED
        if not entry.get(lang)
    ]
    assert missing == [], f"untranslated strings: {missing}"


def test_no_spanish_string_is_a_copy_of_the_english_one():
    # A few labels legitimately coincide; everything else must be translated.
    shared = {"tool_running"}
    copies = [
        key
        for key, entry in STRINGS.items()
        if key not in shared and entry["en"] == entry["es"]
    ]
    assert copies == [], f"strings left untranslated: {copies}"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("es", "es"),
        ("ES", "es"),
        ("es_ES.UTF-8", "es"),
        ("es-419", "es"),
        ("en_US.UTF-8", "en"),
        ("fr_FR", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_maps_locales_to_language_codes(value, expected):
    assert normalize(value) == expected


def test_detect_language_reads_the_environment(monkeypatch):
    for var in ("AGENT_LANG", "LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        monkeypatch.delenv(var, raising=False)
    assert detect_language() == FALLBACK

    monkeypatch.setenv("LANG", "es_ES.UTF-8")
    assert detect_language() == "es"

    # An explicit override wins over the locale.
    monkeypatch.setenv("AGENT_LANG", "en")
    assert detect_language() == "en"


def test_detect_language_ignores_unsupported_locales(monkeypatch):
    for var in ("AGENT_LANG", "LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LANG", "de_DE.UTF-8")
    assert detect_language() == FALLBACK


def test_translator_returns_the_active_language():
    t = Translator("es")
    assert t("goodbye") == "Hasta luego."
    assert t.set_language("en") is True
    assert t("goodbye") == "Bye."


def test_translator_rejects_an_unsupported_language():
    t = Translator("en")
    assert t.set_language("de") is False
    assert t.lang == "en"


def test_translator_interpolates():
    t = Translator("es")
    assert "42" in t("max_steps", n=42)


def test_unknown_keys_degrade_to_the_key_itself():
    assert Translator("es")("nonexistent_key") == "nonexistent_key"
