"""
Deterministic Interview Controller
-----------------------------------
This module owns ALL interview-flow decisions: retries, hints, difficulty
progression, when to move to the next skill/topic/step, and when to
terminate. It does NOT call any LLM. It only reads the interview_plan and
the EvaluationReport produced by Agent 6.

WHY THIS EXISTS
Previously, Agent 5 (an LLM) was asked to "maintain retry count", "maintain
difficulty", "decide when to move on" etc. purely via prompt instructions,
using conversation history and a state blob as its only guide. LLMs are
not reliable at exact multi-turn counting/branching logic — which is
exactly why the bot kept re-asking the same personal-info question forever
in your test run. Moving that logic into plain Python makes it 100%
deterministic and testable. Agent 5's job shrinks to: "given this decision,
phrase the message naturally" — it no longer decides anything.

RULES IMPLEMENTED (per your spec)
Step 1 - Personal Information:
    - Wrong / irrelevant answer on the 1st attempt -> retry the SAME
      question, WITH a hint.
    - Still wrong on the 2nd attempt -> move on to the NEXT question
      regardless of correctness (never ask the same question a 3rd time).

Step 2 / 3 / 4 - Programming Languages / Libraries & Frameworks / Tools
(skill-based steps):
    - Wrong answer on a basic question -> NO hint. Ask a DIFFERENT basic
      question on the SAME skill (related to the one just asked).
    - This repeats up to the planner's allocated `basic_questions` budget
      for that skill.
    - If ALL basic questions for a skill are answered wrong (0 correct)
      AND the skill is mandatory AND terminate_if_failed=True ->
      TERMINATE the interview immediately.
    - As soon as one answer is correct -> move up to the next difficulty
      tier (intermediate, then advanced) for that skill. Whether the next
      question leans "easier" or "harder" within the new tier is driven by
      how technically/logically correct the last answer was.
    - Same tier-exhaustion logic applies to intermediate/advanced, except
      exhausting those tiers without a correct answer does NOT terminate
      the interview (only basic-tier total failure on a mandatory skill
      does) — it just moves on to the next skill.
    - When a skill is completed (all tiers done/exhausted), move to the
      next skill in the step. When all skills in the step are done, move
      to the next step.

Step 5 - Projects:
    - First question is always fixed: "Please explain your project."
    - Vague/irrelevant answers get exactly one retry with a hint, like
      Step 1, then the controller moves on regardless.
    - Bounded by the planner's estimated_questions for this step.

Step 6 - Reasoning:
    - Same one-retry-with-hint pattern as Step 1, plus difficulty
      leaning ("easier"/"harder") for the next reasoning topic based on
      whether the last answer was correct.

Cross-cutting:
    - Offensive language increments a warning counter; a second offense
      terminates the interview.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

TIERS = ["basic", "intermediate", "advanced"]

CORRECT = "correct"
PARTIAL = "partial"
WRONG = "wrong"


def classify(evaluation_report) -> str:
    """Turn Agent 6's EvaluationReport into one of correct/partial/wrong."""
    if evaluation_report is None:
        return WRONG
    if not getattr(evaluation_report, "relevant_answer", False):
        return WRONG
    tech = getattr(evaluation_report, "technically_correct", False)
    logic = getattr(evaluation_report, "logically_correct", False)
    if tech and logic:
        return CORRECT
    if tech or logic:
        return PARTIAL
    return WRONG


@dataclass
class SkillTracker:
    skill_name: str
    mandatory: bool
    terminate_if_failed: bool
    basic_budget: int
    intermediate_budget: int
    advanced_budget: int
    tier: str = "basic"
    asked: Dict[str, int] = field(default_factory=lambda: {"basic": 0, "intermediate": 0, "advanced": 0})
    correct: Dict[str, int] = field(default_factory=lambda: {"basic": 0, "intermediate": 0, "advanced": 0})
    done: bool = False
    failed: bool = False

    def budget_for(self, tier: str) -> int:
        return {"basic": self.basic_budget, "intermediate": self.intermediate_budget,
                "advanced": self.advanced_budget}[tier]

    def next_tier_with_budget(self) -> Optional[str]:
        idx = TIERS.index(self.tier)
        for nxt in TIERS[idx + 1:]:
            if self.budget_for(nxt) > 0:
                return nxt
        return None

    def advance_or_finish(self):
        nxt = self.next_tier_with_budget()
        if nxt:
            self.tier = nxt
        else:
            self.done = True


def build_skill_trackers(skill_plans: List[dict]) -> List[SkillTracker]:
    trackers = []
    for sp in skill_plans or []:
        trackers.append(SkillTracker(
            skill_name=sp["skill_name"],
            mandatory=sp.get("mandatory", False),
            terminate_if_failed=sp.get("terminate_if_failed", False),
            basic_budget=max(sp.get("basic_questions", 0), 0),
            intermediate_budget=max(sp.get("intermediate_questions", 0), 0),
            advanced_budget=max(sp.get("advanced_questions", 0), 0),
        ))
        # If basic budget is 0 for some reason, start at the first tier that
        # actually has a budget so we don't get stuck asking 0 questions.
        if trackers[-1].basic_budget == 0:
            nxt = trackers[-1].next_tier_with_budget()
            trackers[-1].tier = nxt if nxt else "basic"
            if not nxt:
                trackers[-1].done = True
    return trackers


class InterviewController:
    def __init__(self, interview_plan: dict):
        self.plan = interview_plan
        self.step_plans = {sp["step_number"]: sp for sp in interview_plan.get("step_plan", [])}

        # Step 1 - Personal Information
        self.personal_topics = self._topics_for(1, default=["Introduction and Background"])
        self.personal_index = 0
        self.personal_attempt = 1  # 1 = first try, 2 = retry-with-hint already given

        # Steps 2/3/4 - skill based
        self.skill_lists = {
            2: build_skill_trackers(interview_plan.get("programming_languages", [])),
            3: build_skill_trackers(interview_plan.get("libraries_frameworks", [])),
            4: build_skill_trackers(interview_plan.get("tools", [])),
        }
        self.skill_index = {2: 0, 3: 0, 4: 0}

        # Step 5 - Projects
        self.project_topics = interview_plan.get("project_topics", [])
        self.project_budget = self._estimated_questions_for(5, default=3)
        self.project_asked = 0
        self.project_attempt = 1

        # Step 6 - Reasoning
        self.reasoning_topics = interview_plan.get("reasoning_topics", []) or ["General problem solving"]
        self.reasoning_budget = self._estimated_questions_for(6, default=len(self.reasoning_topics))
        self.reasoning_index = 0
        self.reasoning_asked = 0
        self.reasoning_attempt = 1

        self.current_step = 1
        self.terminated = False
        self.completed = False
        self.termination_reason = None
        self.warning_count = 0

    # ---- helpers -----------------------------------------------------

    def _topics_for(self, step_number: int, default: List[str]) -> List[str]:
        sp = self.step_plans.get(step_number)
        if sp and sp.get("topics"):
            return sp["topics"]
        return default

    def _estimated_questions_for(self, step_number: int, default: int) -> int:
        sp = self.step_plans.get(step_number)
        if sp and sp.get("estimated_questions"):
            return sp["estimated_questions"]
        return default

    def _directive(self, action: str, **kwargs) -> Dict[str, Any]:
        d = {
            "action": action,
            "step_number": self.current_step,
            "step_name": self._step_name(self.current_step),
            "skill_name": None,
            "topic": None,
            "tier": None,
            "give_hint": False,
            "difficulty_lean": None,
            "terminate": False,
            "termination_reason": None,
            "complete": False,
            "issue_warning": self.warning_count == 1,
        }
        d.update(kwargs)
        return d

    def _step_name(self, step_number: int) -> str:
        names = {
            1: "Personal Information",
            2: "Programming Languages",
            3: "Libraries / Frameworks",
            4: "Tools",
            5: "Projects",
            6: "Reasoning",
        }
        return names.get(step_number, "Unknown")

    def _advance_step(self):
        self.current_step += 1
        if self.current_step > 6:
            self.completed = True

    def _register_warning(self, last_eval) -> bool:
        """Returns True if this offense just triggered termination."""
        if last_eval is not None and getattr(last_eval, "offensive_language", False):
            self.warning_count += 1
            if self.warning_count >= 2:
                self.terminated = True
                self.termination_reason = "Offensive behaviour continued after a warning."
                return True
        return False

    # ---- main entry point ---------------------------------------------

    def next_action(self, last_evaluation=None) -> Dict[str, Any]:
        if self.terminated:
            return self._directive("TERMINATE", terminate=True, termination_reason=self.termination_reason)
        if self._register_warning(last_evaluation):
            return self._directive("TERMINATE", terminate=True, termination_reason=self.termination_reason)

        if self.current_step == 1:
            return self._handle_personal(last_evaluation)
        elif self.current_step in (2, 3, 4):
            return self._handle_skill_step(last_evaluation)
        elif self.current_step == 5:
            return self._handle_project(last_evaluation)
        elif self.current_step == 6:
            return self._handle_reasoning(last_evaluation)
        else:
            self.completed = True
            return self._directive("INTERVIEW_COMPLETE", complete=True)

    # ---- Step 1: Personal Information -----------------------------------

    def _handle_personal(self, last_eval):
        if last_eval is not None:
            correctness = classify(last_eval)
            if correctness == WRONG and self.personal_attempt == 1:
                self.personal_attempt = 2
                return self._directive(
                    "RETRY_PERSONAL_WITH_HINT",
                    topic=self.personal_topics[self.personal_index],
                    give_hint=True,
                )
            # either answered well, or this was already a retry -> move on
            self.personal_index += 1
            self.personal_attempt = 1

        if self.personal_index >= len(self.personal_topics):
            self._advance_step()
            return self.next_action(None)

        return self._directive(
            "ASK_PERSONAL_QUESTION",
            topic=self.personal_topics[self.personal_index],
        )

    # ---- Steps 2/3/4: skill-based -------------------------------------

    def _handle_skill_step(self, last_eval):
        step = self.current_step
        trackers = self.skill_lists[step]
        idx = self.skill_index[step]

        if idx >= len(trackers):
            self._advance_step()
            return self.next_action(None)

        tracker = trackers[idx]
        difficulty_lean = None

        if last_eval is not None and not tracker.done and not tracker.failed:
            correctness = classify(last_eval)
            tier = tracker.tier

            if correctness in (CORRECT, PARTIAL):
                tracker.correct[tier] += 1
                difficulty_lean = "harder" if correctness == CORRECT else "easier"
                tracker.advance_or_finish()
            else:
                # wrong: check if this tier's budget is exhausted
                if tracker.asked[tier] >= tracker.budget_for(tier):
                    if (
                        tier == "basic"
                        and tracker.correct["basic"] == 0
                        and tracker.mandatory
                        and tracker.terminate_if_failed
                    ):
                        self.terminated = True
                        self.termination_reason = (
                            f"Candidate failed all {tracker.budget_for('basic')} basic "
                            f"questions for mandatory skill '{tracker.skill_name}'."
                        )
                        return self._directive(
                            "TERMINATE", terminate=True, termination_reason=self.termination_reason
                        )
                    else:
                        tracker.advance_or_finish()
                # else: still within budget -> stay on same tier, will ask
                # another question of the same tier below.

        if tracker.done or tracker.failed:
            self.skill_index[step] += 1
            return self.next_action(None)

        tier = tracker.tier
        tracker.asked[tier] += 1
        is_repeat = tracker.asked[tier] > 1
        action = f"ASK_ANOTHER_{tier.upper()}_SAME_SKILL" if is_repeat else f"ASK_{tier.upper()}_QUESTION"

        return self._directive(
            action,
            skill_name=tracker.skill_name,
            tier=tier,
            give_hint=False,  # per spec: no hints in skill steps, just a different question
            difficulty_lean=difficulty_lean,
        )

    # ---- Step 5: Projects -----------------------------------------------

    def _handle_project(self, last_eval):
        if last_eval is not None:
            correctness = classify(last_eval)
            if not getattr(last_eval, "relevant_answer", False) and self.project_attempt == 1:
                self.project_attempt = 2
                return self._directive("RETRY_PROJECT_WITH_HINT", give_hint=True)
            self.project_asked += 1
            self.project_attempt = 1

        if self.project_asked >= self.project_budget:
            self._advance_step()
            return self.next_action(None)

        if self.project_asked == 0:
            return self._directive("PROJECT_INTRO")  # "Please explain your project."

        topic_hint = self.project_topics[self.project_asked % max(len(self.project_topics), 1)] \
            if self.project_topics else None
        return self._directive("PROJECT_FOLLOWUP", topic=topic_hint)

    # ---- Step 6: Reasoning ------------------------------------------------

    def _handle_reasoning(self, last_eval):
        difficulty_lean = None
        if last_eval is not None:
            correctness = classify(last_eval)
            if correctness == WRONG and self.reasoning_attempt == 1:
                self.reasoning_attempt = 2
                return self._directive(
                    "RETRY_REASONING_WITH_HINT",
                    topic=self._current_reasoning_topic(),
                    give_hint=True,
                )
            difficulty_lean = "harder" if correctness == CORRECT else "easier"
            self.reasoning_asked += 1
            self.reasoning_attempt = 1
            self.reasoning_index += 1

        if self.reasoning_asked >= self.reasoning_budget or self.reasoning_index >= len(self.reasoning_topics):
            self._advance_step()
            return self.next_action(None)

        return self._directive(
            "ASK_REASONING_QUESTION",
            topic=self._current_reasoning_topic(),
            difficulty_lean=difficulty_lean,
        )

    def _current_reasoning_topic(self) -> str:
        if not self.reasoning_topics:
            return "General problem solving"
        return self.reasoning_topics[self.reasoning_index % len(self.reasoning_topics)]