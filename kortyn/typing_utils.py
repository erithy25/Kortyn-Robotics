"""Shared type aliases."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[Any]
FloatArray = npt.NDArray[np.float32]
ByteArray = npt.NDArray[np.uint8]
