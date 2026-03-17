"""
sources.py
========================================================================
Source-specific configuration for caption region extraction.

Each broadcast source has a known lower-third layout (position, size,
background colour). The SOURCE_REGISTRY maps source_id strings to
SourceConfig instances that drive the OCR-based caption pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PreScreen:
    """Fast colour check applied to the caption crop before OCR."""
    bg_colour: tuple[int, int, int]   # expected background RGB
    tolerance: int = 30               # per-channel tolerance


@dataclass
class CaptionRegion:
    """
    Normalised coordinates (0.0–1.0) of the caption lower-third.
    Applied as: pixel_x = x * frame_width, etc.
    """
    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        for name, val in [("x", self.x), ("y", self.y), ("w", self.w), ("h", self.h)]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"CaptionRegion.{name} must be in [0.0, 1.0], got {val}"
                )
        if self.x + self.w > 1.0:
            raise ValueError(
                f"CaptionRegion x+w={self.x + self.w:.3f} exceeds 1.0"
            )
        if self.y + self.h > 1.0:
            raise ValueError(
                f"CaptionRegion y+h={self.y + self.h:.3f} exceeds 1.0"
            )


@dataclass
class SourceConfig:
    source_id: str
    caption_region: CaptionRegion
    name_line_index: int = 0          # which OCR line holds the speaker name
    pre_screen: Optional[PreScreen] = None
    notes: str = ""


SOURCE_REGISTRY: dict[str, SourceConfig] = {
    "bbc_politics_live": SourceConfig(
        source_id="bbc_politics_live",
        caption_region=CaptionRegion(x=0.05, y=0.78, w=0.45, h=0.12),
        name_line_index=0,
        pre_screen=PreScreen(bg_colour=(255, 255, 255), tolerance=30),
        notes="White bg, red left-edge bar, bold black name on line 0, party on line 1",
    ),
    "default": SourceConfig(
        source_id="default",
        caption_region=CaptionRegion(x=0.02, y=0.75, w=0.55, h=0.18),
        name_line_index=0,
        pre_screen=None,
        notes="Generic fallback region — no colour pre-screen",
    ),
}

# Validate all registry entries at import time.
# CaptionRegion raises ValueError in __post_init__ if coords are out of range.
# This block is a no-op for valid configs but catches bad hand-edits immediately.
for _k, _cfg in SOURCE_REGISTRY.items():
    _ = _cfg.caption_region  # __post_init__ already ran; just documents intent


def get_source_config(source_id: Optional[str]) -> SourceConfig:
    """Return config for source_id, falling back to 'default'."""
    if source_id and source_id in SOURCE_REGISTRY:
        return SOURCE_REGISTRY[source_id]
    return SOURCE_REGISTRY["default"]
