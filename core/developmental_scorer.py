"""
Developmental Assessment Scoring Engine

Compares observed behaviors against age-appropriate norms to produce:
  - Developmental age equivalent
  - Risk classification: delayed / at_risk_asd / normal / advanced
  - Domain-specific scores with clinical references
  - Actionable recommendations

Clinical references:
  - CDC 2022 Milestones (2mo-5yr)
  - WHO MGRS Motor Windows
  - M-CHAT-R/F items (autism screening, 16-30mo)
  - ASQ-3 domain cut-offs
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════
# Milestone definitions by age (months)
# Each milestone: (description, domain, expected_age_months)
# ═══════════════════════════════════════════════════════════════

@dataclass
class Milestone:
    name: str
    domain: str  # gross_motor, fine_motor, social, language, cognitive
    expected_month: int
    description: str
    how_to_detect: str  # what video feature maps to this


GROSS_MOTOR_MILESTONES = [
    Milestone("head_control", "gross_motor", 4, "Stable head control", "trunk_lean < 20 while prone"),
    Milestone("sitting", "gross_motor", 6, "Sits independently", "stable com, no support"),
    Milestone("crawling", "gross_motor", 9, "Crawling", "alternating limb movement on floor"),
    Milestone("pull_to_stand", "gross_motor", 10, "Pulls to stand", "standing with support"),
    Milestone("walk_with_help", "gross_motor", 12, "Walks with support", "walking with support, step_count > 0"),
    Milestone("walk_alone", "gross_motor", 15, "Walks independently", "walking without support"),
    Milestone("run", "gross_motor", 18, "Runs", "step_freq > 2.5 Hz"),
    Milestone("kick_ball", "gross_motor", 24, "Kicks a ball", "leg swing motion"),
    Milestone("jump_both_feet", "gross_motor", 24, "Jumps with both feet", "both feet off ground"),
    Milestone("climb_stairs", "gross_motor", 24, "Climbs stairs", "alternating stair pattern"),
    Milestone("stand_one_leg_1s", "gross_motor", 30, "Stands on one leg for 1 second", "single_leg_duration >= 1"),
    Milestone("pedal_tricycle", "gross_motor", 36, "Pedals a tricycle", "alternating leg motion"),
    Milestone("hop_one_foot", "gross_motor", 48, "Hops on one foot", "single_leg + jump"),
    Milestone("stand_one_leg_5s", "gross_motor", 60, "Stands on one leg for 5 seconds", "single_leg_duration >= 5"),
]

FINE_MOTOR_MILESTONES = [
    Milestone("reach_grasp", "fine_motor", 5, "Reaches and grasps objects", "hand reaching motion"),
    Milestone("transfer_objects", "fine_motor", 7, "Transfers objects between hands", "object between hands"),
    Milestone("pincer_grasp", "fine_motor", 10, "Pincer grasp", "thumb_index_distance < 0.15"),
    Milestone("put_in_container", "fine_motor", 12, "Puts object in container", "release_count > 0"),
    Milestone("stack_2_blocks", "fine_motor", 15, "Stacks 2 blocks", "placement_count >= 2"),
    Milestone("scribble", "fine_motor", 18, "Scribbles", "wrist movement pattern"),
    Milestone("stack_4_blocks", "fine_motor", 24, "Stacks 4 blocks", "placement_count >= 4"),
    Milestone("turn_pages", "fine_motor", 24, "Turns book pages", "pincer + wrist rotation"),
    Milestone("use_fork", "fine_motor", 30, "Uses a fork", "controlled grasp + arm motion"),
    Milestone("button_clothes", "fine_motor", 48, "Buttons clothing", "fine pincer + coordination"),
    Milestone("write_letters", "fine_motor", 60, "Writes letters", "controlled fine wrist motion"),
]

# M-CHAT-R relevant items that can be assessed via video
SOCIAL_ATTENTION_MILESTONES = [
    Milestone("eye_contact", "social", 3, "Eye contact", "gaze_direction == center, sustained"),
    Milestone("social_smile", "social", 3, "Social smile", "AU6 + AU12 activation"),
    Milestone("follow_gaze", "social", 9, "Follows gaze", "gaze_shift after prompt"),
    Milestone("respond_name", "social", 9, "Responds to name", "head_yaw change on cue"),
    Milestone("point_to_show", "social", 12, "Points to share", "pointing + gaze alternation"),
    Milestone("point_to_request", "social", 12, "Points to request", "pointing toward object"),
    Milestone("imitate_actions", "social", 12, "Imitates actions", "dtw_similarity > 0.5"),
    Milestone("wave_bye", "social", 12, "Waves goodbye", "hand wave pattern"),
    Milestone("show_objects", "social", 15, "Shows objects", "arm extension + gaze to adult"),
    Milestone("nod_yes", "social", 18, "Nods yes", "nod_count > 0"),
    Milestone("shake_no", "social", 18, "Shakes head no", "shake_count > 0"),
    Milestone("pretend_play", "social", 24, "Pretend play", "symbolic object use"),
    Milestone("joint_attention", "social", 12, "Joint attention", "gaze_shift_count >= 3"),
]


# ═══════════════════════════════════════════════════════════════
# M-CHAT-R/F Scoring (Autism screening for 16-30 months)
# Items assessable via video observation
# ═══════════════════════════════════════════════════════════════

MCHAT_VIDEO_ITEMS = {
    "point_follow": {
        "question": "When you point to something across the room, does the child look at it?",
        "detect_key": "gaze_shift_on_point",
        "critical": True,
    },
    "eye_contact": {
        "question": "Does the child make eye contact with you?",
        "detect_key": "sustained_eye_contact",
        "critical": True,
    },
    "imitation": {
        "question": "Does the child imitate your actions (for example, making a face)?",
        "detect_key": "imitation_score",
        "critical": False,
    },
    "pointing": {
        "question": "Does the child point at objects with an index finger?",
        "detect_key": "pointing_detected",
        "critical": True,
    },
    "respond_name": {
        "question": "Does the child turn and look when called by name?",
        "detect_key": "head_turn_on_name",
        "critical": True,
    },
    "show_interest": {
        "question": "Does the child bring objects to show you?",
        "detect_key": "show_behavior",
        "critical": False,
    },
}


# ═══════════════════════════════════════════════════════════════
# Scoring Engine
# ═══════════════════════════════════════════════════════════════

@dataclass
class DomainScore:
    domain: str
    domain_cn: str
    achieved_count: int
    expected_count: int
    score_pct: float  # 0-100
    developmental_age_months: int
    age_difference_months: int  # negative = delayed
    achieved_milestones: list[str] = field(default_factory=list)
    missing_milestones: list[str] = field(default_factory=list)
    advanced_milestones: list[str] = field(default_factory=list)


@dataclass
class ASDScreening:
    items_assessed: int
    items_failed: int
    critical_items_failed: int
    risk_level: str  # low / medium / high
    failed_items: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class DevelopmentalAssessment:
    child_age_months: int
    overall_classification: str  # delayed / at_risk_asd / normal / advanced
    overall_score: float  # 0-100
    domain_scores: list[DomainScore]
    asd_screening: ASDScreening | None
    developmental_age_months: int
    age_equivalent_label: str
    recommendations: list[str]
    disclaimer: str


class DevelopmentalScorer:
    """Accumulates observations and produces developmental assessment."""

    def __init__(self, child_age_months: int):
        self.age = child_age_months
        self.observations: dict[str, float] = {}

    def update_from_gross_motor(self, features: dict):
        if not features:
            return
        self.observations["single_leg_duration"] = max(
            self.observations.get("single_leg_duration", 0),
            features.get("max_single_leg_duration", 0),
        )
        self.observations["step_count"] = max(
            self.observations.get("step_count", 0),
            features.get("step_count", 0),
        )
        self.observations["step_freq"] = max(
            self.observations.get("step_freq", 0),
            features.get("step_frequency_hz", 0),
        )
        self.observations["squat_count"] = max(
            self.observations.get("squat_count", 0),
            features.get("squat_count", 0),
        )
        self.observations["jump_count"] = max(
            self.observations.get("jump_count", 0),
            features.get("jump_count", 0),
        )
        self.observations["trunk_lean"] = features.get("trunk_lean", 0)
        self.observations["com_stability"] = features.get("com_stability", 0)
        arms = features.get("arms_state", "")
        if arms:
            self.observations["arms_raised"] = 1.0

    def update_from_fine_motor(self, features: dict):
        if not features:
            return
        self.observations["grasp_count"] = max(
            self.observations.get("grasp_count", 0),
            features.get("grasp_count", 0),
        )
        self.observations["is_pinching"] = max(
            self.observations.get("is_pinching", 0),
            1.0 if features.get("is_pinching") else 0.0,
        )
        self.observations["release_count"] = max(
            self.observations.get("release_count", 0),
            features.get("release_count", 0),
        )
        self.observations["placement_count"] = max(
            self.observations.get("placement_count", 0),
            features.get("placement_count", 0),
        )

    def update_from_attention(self, features: dict):
        if not features:
            return
        self.observations["gaze_shift_count"] = max(
            self.observations.get("gaze_shift_count", 0),
            features.get("gaze_shift_count", 0),
        )
        self.observations["nod_count"] = max(
            self.observations.get("nod_count", 0),
            features.get("nod_count", 0),
        )
        self.observations["shake_count"] = max(
            self.observations.get("shake_count", 0),
            features.get("shake_count", 0),
        )
        if features.get("is_pointing"):
            self.observations["pointing_detected"] = 1.0
        self.observations["pointing_ratio"] = max(
            self.observations.get("pointing_ratio", 0),
            features.get("pointing_ratio", 0),
        )
        if features.get("gaze_direction") == "center":
            self.observations["eye_contact_frames"] = (
                self.observations.get("eye_contact_frames", 0) + 1
            )
        self.observations["total_gaze_frames"] = (
            self.observations.get("total_gaze_frames", 0) + 1
        )

    def update_from_pain(self, data: dict):
        if not data or data.get("calibrating"):
            return
        aus = data.get("au_activations", {})
        if aus:
            self.observations["nfcs_score"] = data.get("nfcs_score", 0)
            # Track facial expression diversity
            active_aus = sum(1 for v in aus.values() if v > 0.2)
            self.observations["expression_diversity"] = max(
                self.observations.get("expression_diversity", 0),
                active_aus,
            )

    def compute_assessment(self) -> DevelopmentalAssessment:
        gross_score = self._score_gross_motor()
        fine_score = self._score_fine_motor()
        social_score = self._score_social_attention()
        asd = self._screen_asd() if 12 <= self.age <= 36 else None

        domain_scores = [gross_score, fine_score, social_score]
        valid_scores = [d for d in domain_scores if d.expected_count > 0]

        if valid_scores:
            overall = sum(d.score_pct for d in valid_scores) / len(valid_scores)
            dev_ages = [d.developmental_age_months for d in valid_scores]
            dev_age = int(sum(dev_ages) / len(dev_ages))
        else:
            overall = 50.0
            dev_age = self.age

        classification = self._classify(overall, domain_scores, asd)
        age_label = self._age_label(dev_age)
        recommendations = self._generate_recommendations(classification, domain_scores, asd)

        return DevelopmentalAssessment(
            child_age_months=self.age,
            overall_classification=classification,
            overall_score=round(overall, 1),
            domain_scores=domain_scores,
            asd_screening=asd,
            developmental_age_months=dev_age,
            age_equivalent_label=age_label,
            recommendations=recommendations,
            disclaimer=(
                "This assessment is an AI-assisted screening tool and does not constitute a clinical diagnosis. "
                "All results should be reviewed and confirmed by a pediatrician or developmental specialist. "
                "Final classification decisions remain the responsibility of licensed clinicians."
            ),
        )

    # ── Domain scoring ──

    def _score_gross_motor(self) -> DomainScore:
        achieved = []
        missing = []
        advanced = []

        obs = self.observations
        checks = {
            "walk_alone": obs.get("step_count", 0) >= 3,
            "run": obs.get("step_freq", 0) > 2.5,
            "jump_both_feet": obs.get("jump_count", 0) >= 1,
            "stand_one_leg_1s": obs.get("single_leg_duration", 0) >= 1.0,
            "stand_one_leg_5s": obs.get("single_leg_duration", 0) >= 5.0,
            "hop_one_foot": (obs.get("single_leg_duration", 0) >= 0.5 and obs.get("jump_count", 0) >= 1),
            "climb_stairs": obs.get("step_count", 0) >= 5,
        }

        expected_by_age = [
            m for m in GROSS_MOTOR_MILESTONES if m.expected_month <= self.age
        ]
        above_age = [
            m for m in GROSS_MOTOR_MILESTONES if m.expected_month > self.age
        ]

        for m in expected_by_age:
            if m.name in checks and checks[m.name]:
                achieved.append(m.description)
            elif m.name in checks:
                missing.append(m.description)

        for m in above_age:
            if m.name in checks and checks[m.name]:
                advanced.append(m.description)

        expected_count = len([m for m in expected_by_age if m.name in checks])
        achieved_count = len(achieved)
        score_pct = (achieved_count / max(expected_count, 1)) * 100

        dev_age = self._estimate_dev_age(GROSS_MOTOR_MILESTONES, checks)

        return DomainScore(
            domain="gross_motor",
            domain_cn="Gross Motor",
            achieved_count=achieved_count,
            expected_count=expected_count,
            score_pct=round(score_pct, 1),
            developmental_age_months=dev_age,
            age_difference_months=dev_age - self.age,
            achieved_milestones=achieved,
            missing_milestones=missing,
            advanced_milestones=advanced,
        )

    def _score_fine_motor(self) -> DomainScore:
        obs = self.observations
        checks = {
            "pincer_grasp": obs.get("is_pinching", 0) > 0 or obs.get("grasp_count", 0) > 0,
            "put_in_container": obs.get("release_count", 0) >= 1,
            "stack_2_blocks": obs.get("placement_count", 0) >= 2,
            "stack_4_blocks": obs.get("placement_count", 0) >= 4,
        }

        achieved, missing, advanced = [], [], []
        expected_by_age = [m for m in FINE_MOTOR_MILESTONES if m.expected_month <= self.age]
        above_age = [m for m in FINE_MOTOR_MILESTONES if m.expected_month > self.age]

        for m in expected_by_age:
            if m.name in checks and checks[m.name]:
                achieved.append(m.description)
            elif m.name in checks:
                missing.append(m.description)

        for m in above_age:
            if m.name in checks and checks[m.name]:
                advanced.append(m.description)

        expected_count = len([m for m in expected_by_age if m.name in checks])
        achieved_count = len(achieved)
        score_pct = (achieved_count / max(expected_count, 1)) * 100
        dev_age = self._estimate_dev_age(FINE_MOTOR_MILESTONES, checks)

        return DomainScore(
            domain="fine_motor",
            domain_cn="Fine Motor",
            achieved_count=achieved_count,
            expected_count=expected_count,
            score_pct=round(score_pct, 1),
            developmental_age_months=dev_age,
            age_difference_months=dev_age - self.age,
            achieved_milestones=achieved,
            missing_milestones=missing,
            advanced_milestones=advanced,
        )

    def _score_social_attention(self) -> DomainScore:
        obs = self.observations
        total_gaze = obs.get("total_gaze_frames", 0)
        eye_contact_ratio = (
            obs.get("eye_contact_frames", 0) / max(total_gaze, 1)
        )

        checks = {
            "eye_contact": eye_contact_ratio > 0.2,
            "follow_gaze": obs.get("gaze_shift_count", 0) >= 2,
            "point_to_show": obs.get("pointing_detected", 0) > 0,
            "point_to_request": obs.get("pointing_ratio", 0) > 0.05,
            "imitate_actions": False,  # needs dual-video
            "nod_yes": obs.get("nod_count", 0) >= 2,
            "shake_no": obs.get("shake_count", 0) >= 2,
            "joint_attention": obs.get("gaze_shift_count", 0) >= 3,
        }

        achieved, missing, advanced = [], [], []
        expected_by_age = [m for m in SOCIAL_ATTENTION_MILESTONES if m.expected_month <= self.age]
        above_age = [m for m in SOCIAL_ATTENTION_MILESTONES if m.expected_month > self.age]

        for m in expected_by_age:
            if m.name in checks and checks[m.name]:
                achieved.append(m.description)
            elif m.name in checks:
                missing.append(m.description)

        for m in above_age:
            if m.name in checks and checks[m.name]:
                advanced.append(m.description)

        expected_count = len([m for m in expected_by_age if m.name in checks])
        achieved_count = len(achieved)
        score_pct = (achieved_count / max(expected_count, 1)) * 100
        dev_age = self._estimate_dev_age(SOCIAL_ATTENTION_MILESTONES, checks)

        return DomainScore(
            domain="social_attention",
            domain_cn="Social / Attention",
            achieved_count=achieved_count,
            expected_count=expected_count,
            score_pct=round(score_pct, 1),
            developmental_age_months=dev_age,
            age_difference_months=dev_age - self.age,
            achieved_milestones=achieved,
            missing_milestones=missing,
            advanced_milestones=advanced,
        )

    def _screen_asd(self) -> ASDScreening:
        obs = self.observations
        total_gaze = obs.get("total_gaze_frames", 0)
        eye_contact_ratio = obs.get("eye_contact_frames", 0) / max(total_gaze, 1)

        item_results = {}
        # Each item: True = PASS (no concern), False = FAIL (concern)
        item_results["eye_contact"] = eye_contact_ratio > 0.15
        item_results["pointing"] = obs.get("pointing_detected", 0) > 0
        item_results["point_follow"] = obs.get("gaze_shift_count", 0) >= 2
        item_results["imitation"] = False  # can't assess without reference
        item_results["respond_name"] = obs.get("nod_count", 0) + obs.get("shake_count", 0) > 0

        assessed = len(item_results)
        failed = [k for k, v in item_results.items() if not v]
        critical_failed = [
            k for k in failed if k in MCHAT_VIDEO_ITEMS and MCHAT_VIDEO_ITEMS[k].get("critical")
        ]

        if len(critical_failed) >= 2 or len(failed) >= 3:
            risk = "high"
        elif len(critical_failed) >= 1 or len(failed) >= 2:
            risk = "medium"
        else:
            risk = "low"

        failed_descriptions = []
        for k in failed:
            if k in MCHAT_VIDEO_ITEMS:
                failed_descriptions.append(MCHAT_VIDEO_ITEMS[k]["question"])

        note = ""
        if "imitation" in failed:
            note = "Imitation requires dual-video comparison and cannot be fully assessed with a single camera feed."

        return ASDScreening(
            items_assessed=assessed,
            items_failed=len(failed),
            critical_items_failed=len(critical_failed),
            risk_level=risk,
            failed_items=failed_descriptions,
            note=note,
        )

    # ── Helpers ──

    def _estimate_dev_age(self, milestones: list[Milestone], checks: dict) -> int:
        """Find the highest milestone age that was achieved."""
        achieved_ages = []
        for m in milestones:
            if m.name in checks and checks[m.name]:
                achieved_ages.append(m.expected_month)
        if achieved_ages:
            return max(achieved_ages)
        # If nothing achieved, estimate based on lowest expected
        lowest = [m.expected_month for m in milestones if m.name in checks]
        if lowest:
            return max(0, min(lowest) - 3)
        return max(0, self.age - 6)

    def _classify(
        self, overall: float, domains: list[DomainScore], asd: ASDScreening | None
    ) -> str:
        # ASD risk takes priority if high
        if asd and asd.risk_level == "high":
            return "at_risk_asd"

        # Check for significant delay (any domain >6 months behind)
        severe_delay = any(d.age_difference_months <= -6 for d in domains if d.expected_count > 0)
        if severe_delay:
            return "significant_delay"

        # Moderate delay
        moderate_delay = any(d.age_difference_months <= -3 for d in domains if d.expected_count > 0)
        if moderate_delay or overall < 40:
            return "mild_delay"

        # ASD medium risk
        if asd and asd.risk_level == "medium":
            return "at_risk_asd"

        # Advanced
        advanced = any(len(d.advanced_milestones) >= 2 for d in domains)
        if advanced and overall >= 80:
            return "advanced"

        if overall >= 60:
            return "normal"

        return "borderline"

    def _age_label(self, dev_age: int) -> str:
        if dev_age <= 0:
            return "0-3 month level"
        if dev_age == self.age:
            return f"Age-appropriate at {self.age} months"
        if dev_age > self.age:
            return f"Approximately {dev_age}-month level ({dev_age - self.age} months ahead)"
        return f"Approximately {dev_age}-month level ({self.age - dev_age} months behind)"

    def _generate_recommendations(
        self, classification: str, domains: list[DomainScore], asd: ASDScreening | None
    ) -> list[str]:
        recs = []

        if classification == "significant_delay":
            recs.append("Significant developmental delay detected. Please arrange a specialist developmental evaluation as soon as possible.")
            for d in domains:
                if d.age_difference_months <= -6 and d.missing_milestones:
                    recs.append(f"  Missing milestones in {d.domain_cn}: {', '.join(d.missing_milestones[:3])}")

        elif classification == "mild_delay":
            recs.append("Some domains are progressing more slowly than expected. Discuss these findings with your pediatrician at the next follow-up.")
            for d in domains:
                if d.age_difference_months <= -3 and d.missing_milestones:
                    recs.append(f"  {d.domain_cn}: Monitor {', '.join(d.missing_milestones[:3])}")

        elif classification == "at_risk_asd":
            recs.append("Social/attention indicators suggest possible ASD risk. A full professional M-CHAT-R/F screening is recommended.")
            if asd and asd.failed_items:
                recs.append("  Screening items not passed:")
                for item in asd.failed_items[:4]:
                    recs.append(f"    • {item}")
            recs.append("  Recommendation: Contact child psychiatry or developmental-behavioral pediatrics early for a comprehensive evaluation.")

        elif classification == "advanced":
            recs.append("Development appears advanced for age. Consider offering richer and age-appropriate learning challenges.")
            for d in domains:
                if d.advanced_milestones:
                    recs.append(f"  {d.domain_cn}: Already achieved {', '.join(d.advanced_milestones[:3])}")

        elif classification == "normal":
            recs.append("Developmental indicators are within the expected range. Continue routine pediatric follow-up.")

        elif classification == "borderline":
            recs.append("Borderline profile: some indicators are near threshold. Consider re-screening in 1-2 months.")

        # Domain-specific tips
        for d in domains:
            if d.score_pct < 50 and d.expected_count > 0:
                if d.domain == "gross_motor":
                    recs.append(f"Tip - {d.domain_cn}: Increase outdoor play and practice running, jumping, and climbing.")
                elif d.domain == "fine_motor":
                    recs.append(f"Tip - {d.domain_cn}: Use fine-motor games such as blocks, bead threading, and paper tearing.")
                elif d.domain == "social_attention":
                    recs.append(f"Tip - {d.domain_cn}: Increase face-to-face interaction games and practice pointing, imitation, and turn-taking.")

        return recs


def assessment_to_dict(a: DevelopmentalAssessment) -> dict:
    """Convert assessment to JSON-serializable dict."""
    return {
        "child_age_months": a.child_age_months,
        "overall_classification": a.overall_classification,
        "overall_score": a.overall_score,
        "developmental_age_months": a.developmental_age_months,
        "age_equivalent_label": a.age_equivalent_label,
        "domain_scores": [
            {
                "domain": d.domain,
                "domain_cn": d.domain_cn,
                "achieved_count": d.achieved_count,
                "expected_count": d.expected_count,
                "score_pct": d.score_pct,
                "developmental_age_months": d.developmental_age_months,
                "age_difference_months": d.age_difference_months,
                "achieved_milestones": d.achieved_milestones,
                "missing_milestones": d.missing_milestones,
                "advanced_milestones": d.advanced_milestones,
            }
            for d in a.domain_scores
        ],
        "asd_screening": {
            "items_assessed": a.asd_screening.items_assessed,
            "items_failed": a.asd_screening.items_failed,
            "critical_items_failed": a.asd_screening.critical_items_failed,
            "risk_level": a.asd_screening.risk_level,
            "failed_items": a.asd_screening.failed_items,
            "note": a.asd_screening.note,
        } if a.asd_screening else None,
        "recommendations": a.recommendations,
        "disclaimer": a.disclaimer,
    }
