"""The pinned numbers the tests read (tests/pins.yaml). `from pins import PINS`."""
from pathlib import Path

import yaml

PINS = yaml.safe_load((Path(__file__).with_name("pins.yaml")).read_text())
