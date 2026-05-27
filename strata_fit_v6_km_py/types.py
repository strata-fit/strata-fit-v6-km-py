from enum import Enum


class EventType(str, Enum):
    EXACT = "exact"
    CENSORED = "censored"
    INTERVAL = "interval"


class NoiseType(str, Enum):
    NONE = "NONE"
    GAUSSIAN = "GAUSSIAN"
    POISSON = "POISSON"

    @classmethod
    def coerce(cls, value: "NoiseType | str | None") -> "NoiseType":
        if value is None or value == "":
            return cls.NONE
        if isinstance(value, cls):
            return value
        return cls(str(value).upper())

# Hyperparameters for column names.
# After preprocessing, the survival data will always have these standardized column names.
DEFAULT_INTERVAL_START_COLUMN = "interval_start"
DEFAULT_INTERVAL_END_COLUMN = "interval_end"
DEFAULT_EVENT_INDICATOR_COLUMN = "event_type"
DEFAULT_CUMULATIVE_INCIDENCE_COLUMN = "cumulative_incidence"
MINIMUM_ORGANIZATIONS = 3
