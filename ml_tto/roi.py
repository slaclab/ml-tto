"""Region geometry used by beam-size validation."""

import numpy as np
from pydantic import BaseModel, ConfigDict, PositiveFloat, field_validator


class ROI(BaseModel):
    model_config = ConfigDict(extra="forbid")

    center: tuple[float, float]
    extent: tuple[PositiveFloat, PositiveFloat]


class CircularROI(BaseModel):
    model_config = ConfigDict(extra="forbid")

    center: tuple[float, float]
    radius: tuple[PositiveFloat]

    @field_validator("radius", mode="before")
    @classmethod
    def normalize_radius(cls, value):
        if np.isscalar(value):
            return (value,)
        return value
