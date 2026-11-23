"""Roborock Custom Map integration."""

from __future__ import annotations

import shutil

from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import (
    CONF_BG_OVERRIDES,
    CONF_MAP_COLORS,
    CONF_MAP_ROTATION,
    DATA_LAST_BG_OVERRIDES,
    DATA_LAST_MAP_COLORS,
    DATA_PREVIEW_SESSIONS,
    DOMAIN,
    signal_map_refresh,
)
from .map_render import clear_scaled_backgrounds, override_dir

PLATFORMS = [Platform.IMAGE, Platform.SELECT]


def _overrides_snapshot(entry: ConfigEntry) -> dict[str, dict]:
    return {
        key: dict(record)
        for key, record in entry.options.get(CONF_BG_OVERRIDES, {}).items()
    }


def _colors_snapshot(entry: ConfigEntry) -> dict[str, dict]:
    return {
        key: dict(record)
        for key, record in entry.options.get(CONF_MAP_COLORS, {}).items()
    }


def _changed_keys(old: dict | None, new: dict) -> set[str]:
    if old is None:
        return set(new)
    return {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Roborock Custom map from a config entry."""
    roborock_entries = hass.config_entries.async_entries("roborock")
    coordinators = []

    @callback
    def unload_this_entry() -> None:
        hass.async_create_task(hass.config_entries.async_reload(entry.entry_id))

    for r_entry in roborock_entries:
        if r_entry.state == ConfigEntryState.LOADED:
            coordinators.extend(r_entry.runtime_data.v1)
            r_entry.async_on_unload(unload_this_entry)

    if not coordinators:
        raise ConfigEntryNotReady("No Roborock entries loaded. Cannot start.")

    entry.runtime_data = coordinators

    data = hass.data.setdefault(DOMAIN, {}).setdefault(entry.entry_id, {})
    data.setdefault(CONF_MAP_ROTATION, {})
    data.setdefault(DATA_PREVIEW_SESSIONS, {})
    data[DATA_LAST_BG_OVERRIDES] = _overrides_snapshot(entry)
    data[DATA_LAST_MAP_COLORS] = _colors_snapshot(entry)

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Refresh only the maps whose background override or colors changed."""
    data = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    new_bg = _overrides_snapshot(entry)
    new_colors = _colors_snapshot(entry)
    old_bg = data.get(DATA_LAST_BG_OVERRIDES) if data is not None else None
    old_colors = data.get(DATA_LAST_MAP_COLORS) if data is not None else None
    if data is not None:
        data[DATA_LAST_BG_OVERRIDES] = new_bg
        data[DATA_LAST_MAP_COLORS] = new_colors

    changed = _changed_keys(old_bg, new_bg) | _changed_keys(old_colors, new_colors)
    for key in changed:
        async_dispatcher_send(hass, signal_map_refresh(entry.entry_id, key))


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
        clear_scaled_backgrounds()
    return unloaded


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Clean up stored background override files when the last entry is removed."""
    if hass.config_entries.async_entries(DOMAIN):
        return
    await hass.async_add_executor_job(
        lambda: shutil.rmtree(override_dir(hass), ignore_errors=True)
    )
