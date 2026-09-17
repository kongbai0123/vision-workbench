"""Source-checkout compatibility path for the canonical src package."""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / 'src' / 'vision_workbench')]
