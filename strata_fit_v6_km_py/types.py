from enum import Enum

class EventType(str, Enum):
    EXACT = "exact"
    CENSORED = "censored"
    INTERVAL = "interval"

class NoiseType(str, Enum):
    NONE = "NONE"
    GAUSSIAN = "GAUSSIAN"
    POISSON = "POISSON"

# Hyperparameters for column names.
# After preprocessing, the survival data will always have these standardized column names.
DEFAULT_INTERVAL_START_COLUMN = "interval_start"
DEFAULT_INTERVAL_END_COLUMN = "interval_end"
DEFAULT_EVENT_INDICATOR_COLUMN = "event_type"
DEFAULT_CUMULATIVE_INCIDENCE_COLUMN = "cumulative_incidence"
DEFAULT_EVER_D2T_COLUMN = "ever_d2t"
DEFAULT_FIRST_D2T_MONTH_COLUMN = "first_d2t_month"
DEFAULT_D2T_STEP3_COLUMN = "D2T_step3"
MINIMUM_ORGANIZATIONS = 3
