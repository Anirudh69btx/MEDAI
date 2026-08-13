"""MindCare AI - Mental Health Recommendation Engine.

Generates evidence-based, personalised mental health recommendations from a
model prediction result. Recommendations are organised into tiers:

1. Crisis / Immediate intervention (high-risk classes)
2. Professional referral suggestions
3. Self-care and lifestyle recommendations
4. Psychoeducation and resources

The engine is intentionally stateless and rule-based so it is transparent,
auditable, and requires no additional ML dependencies.

DISCLAIMER: Recommendations are informational only and must not replace
advice from a qualified mental health professional.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .logger import get_logger

logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
#  Class-to-recommendation knowledge base
# ---------------------------------------------------------------------------
# Keys correspond to lowercase, stripped class names produced by the
# LabelEncoder.  Unknown classes fall back to the ``default`` key.

_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    "depression": {
        "risk_level": "high",
        "immediate_actions": [
            "Reach out to a mental health professional as soon as possible.",
            "If you are in crisis, contact a crisis helpline immediately.",
        ],
        "professional_referral": [
            "Consider scheduling an appointment with a psychiatrist or clinical psychologist.",
            "Cognitive Behavioural Therapy (CBT) has strong evidence for depression.",
        ],
        "self_care": [
            "Maintain a consistent daily routine including regular sleep and meals.",
            "Engage in at least 30 minutes of moderate physical activity daily.",
            "Limit alcohol and avoid substance use.",
        ],
        "resources": [
            "iCall Helpline (India): 9152987821",
            "Vandrevala Foundation: 1860-2662-345",
        ],
    },
    "anxiety": {
        "risk_level": "moderate",
        "immediate_actions": [
            "Practice slow, diaphragmatic breathing to calm the nervous system.",
        ],
        "professional_referral": [
            "Consult a psychologist trained in Cognitive Behavioural Therapy (CBT) or Acceptance and Commitment Therapy (ACT).",
        ],
        "self_care": [
            "Limit caffeine and alcohol intake.",
            "Practice mindfulness meditation for 10–20 minutes daily.",
            "Engage in regular aerobic exercise.",
        ],
        "resources": [
            "iCall Helpline (India): 9152987821",
            "Anxiety and Depression Association of America: https://adaa.org",
        ],
    },
    "stress": {
        "risk_level": "low",
        "immediate_actions": [
            "Take short breaks during work and avoid prolonged screen time.",
        ],
        "professional_referral": [
            "Consider speaking to a counsellor if stress persists beyond two weeks.",
        ],
        "self_care": [
            "Prioritise quality sleep (7–9 hours per night).",
            "Identify and limit stressors where possible.",
            "Practice relaxation techniques such as yoga or progressive muscle relaxation.",
        ],
        "resources": [
            "Vandrevala Foundation: 1860-2662-345",
        ],
    },
    "normal": {
        "risk_level": "minimal",
        "immediate_actions": [],
        "professional_referral": [
            "Continue routine mental health check-ins to maintain wellbeing.",
        ],
        "self_care": [
            "Continue healthy sleep, exercise, and social connection habits.",
            "Practice gratitude journaling to reinforce positive mental states.",
        ],
        "resources": [
            "WHO Mental Health: https://www.who.int/health-topics/mental-health",
        ],
    },
    "yes": {
        "risk_level": "moderate_to_high",
        "immediate_actions": [
            "Schedule a consultation with a licensed mental health professional.",
            "If experiencing acute distress, contact a mental health crisis helpline immediately.",
        ],
        "professional_referral": [
            "Consult a psychiatrist or clinical psychologist for a formal evaluation.",
            "Explore evidence-based therapies such as Cognitive Behavioural Therapy (CBT).",
        ],
        "self_care": [
            "Maintain a regular daily routine including consistent sleep hygiene and meals.",
            "Practice stress-reduction techniques such as mindfulness or structured breathing.",
            "Maintain strong social connections with trusted friends, family, or colleagues.",
        ],
        "resources": [
            "iCall Helpline (India): 9152987821",
            "Vandrevala Foundation: 1860-2662-345",
            "988 Suicide & Crisis Lifeline (US): Call/Text 988",
        ],
    },
    "no": {
        "risk_level": "minimal_to_low",
        "immediate_actions": [],
        "professional_referral": [
            "Continue periodic wellness check-ins with your healthcare provider.",
        ],
        "self_care": [
            "Maintain regular sleep schedules, physical exercise, and balanced nutrition.",
            "Practice proactive stress management and healthy work-life boundaries.",
            "Nurture supportive social relationships.",
        ],
        "resources": [
            "WHO Mental Health: https://www.who.int/health-topics/mental-health",
        ],
    },
    "treatment": {
        "risk_level": "moderate_to_high",
        "immediate_actions": [
            "Schedule a consultation with a licensed mental health professional.",
        ],
        "professional_referral": [
            "Consult a psychiatrist or clinical psychologist for evaluation.",
        ],
        "self_care": [
            "Maintain a regular daily routine and practice stress-reduction techniques.",
        ],
        "resources": [
            "iCall Helpline (India): 9152987821",
        ],
    },
    "no_treatment": {
        "risk_level": "minimal_to_low",
        "immediate_actions": [],
        "professional_referral": [
            "Continue periodic mental health check-ins.",
        ],
        "self_care": [
            "Maintain regular sleep, physical activity, and social connections.",
        ],
        "resources": [
            "WHO Mental Health: https://www.who.int/health-topics/mental-health",
        ],
    },
    "default": {
        "risk_level": "unknown",
        "immediate_actions": [
            "Consult a qualified mental health professional for a thorough assessment.",
        ],
        "professional_referral": [
            "Seek evaluation from a licensed psychiatrist or clinical psychologist.",
        ],
        "self_care": [
            "Maintain regular sleep, balanced nutrition, and physical activity.",
            "Avoid isolation – maintain social connections.",
        ],
        "resources": [
            "iCall Helpline (India): 9152987821",
            "WHO Mental Health: https://www.who.int/health-topics/mental-health",
        ],
    },
}


@dataclass
class RecommendationResult:
    """Structured container for generated recommendations.

    Attributes
    ----------
    predicted_class : str
        The raw predicted class label.
    risk_level : str
        Assessed risk tier: 'minimal', 'low', 'moderate', 'high', or 'unknown'.
    confidence : float
        Model prediction confidence score (probability), range [0, 1].
    immediate_actions : List[str]
        Urgent steps to take immediately.
    professional_referral : List[str]
        Recommendations for professional mental health support.
    self_care : List[str]
        Evidence-based self-care strategies.
    resources : List[str]
        Helpline contacts and educational resources.
    disclaimer : str
        Standard clinical disclaimer text.
    """

    predicted_class: str
    risk_level: str
    confidence: float
    immediate_actions: List[str] = field(default_factory=list)
    professional_referral: List[str] = field(default_factory=list)
    self_care: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    disclaimer: str = (
        "DISCLAIMER: These recommendations are for informational purposes only "
        "and do not constitute medical advice. Please consult a licensed mental "
        "health professional for a proper diagnosis and treatment plan."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "predicted_class": self.predicted_class,
            "risk_level": self.risk_level,
            "confidence": round(self.confidence, 4),
            "immediate_actions": self.immediate_actions,
            "professional_referral": self.professional_referral,
            "self_care": self.self_care,
            "resources": self.resources,
            "disclaimer": self.disclaimer,
        }


class RecommendationEngine:
    """Generates personalised mental health recommendations from model predictions.

    The engine uses a transparent, rule-based knowledge base indexed by
    predicted class label.  Unknown classes receive a safe default response.

    Usage
    -----
    >>> engine = RecommendationEngine()
    >>> result = engine.generate(predicted_class="depression", confidence=0.87)
    >>> print(result.risk_level)
    'high'
    """

    def __init__(
        self,
        knowledge_base: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Initialise the recommendation engine.

        Parameters
        ----------
        knowledge_base : dict, optional
            Custom recommendation knowledge base. Defaults to the built-in
            ``_RECOMMENDATIONS`` dict. Must contain a ``'default'`` key.
        """
        self._kb: Dict[str, Dict[str, Any]] = knowledge_base or _RECOMMENDATIONS

        if "default" not in self._kb:
            raise ValueError("Knowledge base must contain a 'default' fallback key.")

    def generate(
        self,
        predicted_class: str,
        confidence: float = 1.0,
    ) -> RecommendationResult:
        """Generate a ``RecommendationResult`` for the given *predicted_class*.

        Parameters
        ----------
        predicted_class : str
            Class label predicted by the model.
        confidence : float, optional
            Model prediction confidence. Defaults to 1.0.

        Returns
        -------
        RecommendationResult
            Structured recommendation object.
        """
        key = predicted_class.strip().lower()
        entry = self._kb.get(key, self._kb["default"])

        if key not in self._kb:
            logger.warning(
                "Predicted class '%s' not found in knowledge base. Using default.",
                predicted_class,
            )

        result = RecommendationResult(
            predicted_class=predicted_class,
            risk_level=entry.get("risk_level", "unknown"),
            confidence=float(confidence),
            immediate_actions=list(entry.get("immediate_actions", [])),
            professional_referral=list(entry.get("professional_referral", [])),
            self_care=list(entry.get("self_care", [])),
            resources=list(entry.get("resources", [])),
        )

        logger.info(
            "Recommendation generated: class='%s', risk='%s', confidence=%.4f",
            predicted_class,
            result.risk_level,
            confidence,
        )
        return result

    def get_supported_classes(self) -> List[str]:
        """Return all class keys that have explicit entries in the knowledge base.

        Returns
        -------
        List[str]
            Class label strings (excluding the 'default' key).
        """
        return [k for k in self._kb.keys() if k != "default"]


__all__ = ["RecommendationEngine", "RecommendationResult"]
