class BatteryPolicy:
    """
    Defines how inference behaves at a particular
    battery level.
    """

    def __init__(
        self,
        name,
        exit1_threshold,
        exit2_threshold,
        max_exit,
    ):
        self.name = name
        self.exit1_threshold = exit1_threshold
        self.exit2_threshold = exit2_threshold
        self.max_exit = max_exit


# Adaptive battery policies

LOW_BATTERY = BatteryPolicy(
    name="low",
    exit1_threshold=0.70,
    exit2_threshold=None,
    max_exit=2,
)

MEDIUM_BATTERY = BatteryPolicy(
    name="medium",
    exit1_threshold=0.75,
    exit2_threshold=0.70,
    max_exit=3,
)

HIGH_BATTERY = BatteryPolicy(
    name="high",
    exit1_threshold=0.85,
    exit2_threshold=0.75,
    max_exit=3,
)


# Battery policy
def get_battery_policy(battery_level):
    """
    Return the inference policy for the current
    battery percentage.
    """

    if battery_level <= 20:
        return LOW_BATTERY

    if battery_level <= 50:
        return MEDIUM_BATTERY

    return HIGH_BATTERY


