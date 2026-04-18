"""
WHO MGRS + CDC developmental milestone normative data.
Used for percentile comparison — not for training.
"""

# WHO Motor Development Windows (months): {milestone: (5th_pct, 50th_pct, 95th_pct)}
WHO_GROSS_MOTOR_NORMS: dict[str, tuple[float, float, float]] = {
    "sitting_without_support": (3.8, 5.9, 9.2),
    "standing_with_assistance": (4.8, 7.6, 11.4),
    "walking_with_assistance": (5.9, 9.2, 13.7),
    "standing_alone": (6.9, 10.8, 16.9),
    "walking_alone": (8.2, 12.1, 17.6),
    "hands_and_knees_crawling": (5.2, 8.3, 13.5),
}

# CDC 2022 milestones by age (months) → list of expected abilities
CDC_MILESTONES: dict[int, dict[str, list[str]]] = {
    12: {
        "gross_motor": ["pulls up to stand", "walks holding on to furniture"],
        "fine_motor": ["puts things in a container", "picks things up with thumb and one finger"],
        "social_emotional": ["plays games with you like pat-a-cake"],
    },
    18: {
        "gross_motor": ["walks without holding on", "climbs on and off furniture without help"],
        "fine_motor": ["tries to use a spoon", "scribbles"],
        "social_emotional": ["moves away from you but looks to make sure you are nearby"],
    },
    24: {
        "gross_motor": ["kicks a ball", "runs", "walks up a few stairs with or without help"],
        "fine_motor": ["uses things with both hands like large crayons", "turns pages"],
        "social_emotional": ["notices when others are hurt or upset"],
    },
    30: {
        "gross_motor": ["uses hands to twist things", "jumps off the ground with both feet"],
        "fine_motor": ["uses a fork"],
        "social_emotional": ["plays next to other children"],
    },
    36: {
        "gross_motor": ["pedals a tricycle", "runs well without falling"],
        "fine_motor": ["strings items together like large beads", "puts on some clothes by self"],
        "social_emotional": ["calms down within 10 minutes after you leave"],
    },
    48: {
        "gross_motor": ["catches a large ball most of the time", "hops on one foot"],
        "fine_motor": ["unbuttons some buttons", "holds crayon between fingers and thumb"],
        "social_emotional": ["pretends to be something else during play"],
    },
    60: {
        "gross_motor": ["hops on one foot", "stands on one foot for a count of 5"],
        "fine_motor": ["writes some letters in own name", "buttons some buttons"],
        "social_emotional": ["does simple chores at home"],
    },
}

# Single-leg stand duration norms by age (months): (borderline_sec, normal_sec)
SINGLE_LEG_STAND_NORMS: dict[int, tuple[float, float]] = {
    24: (1.0, 2.0),
    30: (1.5, 3.0),
    36: (2.0, 5.0),
    48: (3.0, 8.0),
    60: (5.0, 10.0),
}

# Gait symmetry norms: (borderline_ratio, normal_ratio)  ratio = left/right step time
GAIT_SYMMETRY_NORM = (0.85, 0.92)

# Step frequency norms by age (months): (low, high) Hz
STEP_FREQUENCY_NORMS: dict[int, tuple[float, float]] = {
    18: (1.2, 2.0),
    24: (1.5, 2.5),
    36: (1.8, 2.8),
    48: (2.0, 3.0),
}

# Pincer grasp: thumb-index distance threshold (normalized by hand size)
PINCER_DISTANCE_THRESHOLD = 0.08

# Block stacking stability: max center-of-mass drift ratio
BLOCK_STACK_STABILITY_THRESHOLD = 0.15

# Joint attention latency norms by age (months): (borderline_sec, normal_sec)
JOINT_ATTENTION_LATENCY_NORMS: dict[int, tuple[float, float]] = {
    12: (3.0, 1.5),
    18: (2.5, 1.2),
    24: (2.0, 1.0),
    36: (1.5, 0.8),
}

# NFCS (Neonatal Facial Coding System) thresholds
NFCS_PAIN_THRESHOLD = 3.0
NFCS_BORDERLINE_THRESHOLD = 1.5


def get_closest_age_norm(age_months: int, norms: dict[int, any]) -> any:
    ages = sorted(norms.keys())
    closest = min(ages, key=lambda a: abs(a - age_months))
    return norms[closest]
