from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Optional
import json


# Structured Output Schema

class EvaluationReport(BaseModel):
    relevant_answer: bool = Field(
        description="Whether the candidate answered the asked question."
    )
    technically_correct: bool = Field(
        description="Whether the answer is technically correct."
    )
    logically_correct: bool = Field(
        description="Whether the answer is logically correct."
    )
    offensive_language: bool = Field(
        description="Whether the answer contains offensive or inappropriate language."
    )
    need_warning: bool = Field(
        description="Whether the candidate should receive a warning."
    )
    need_hint: bool = Field(
        description="Whether the next question should include a hint."
    )
    retry_question: bool = Field(
        description="Whether the same question should be asked again."
    )
    candidate_understands_topic: bool = Field(
        description="Whether the candidate demonstrates understanding of the topic."
    )
    confidence_score: float = Field(
        description="Confidence score between 0 and 100."
    )
    answer_quality: str = Field(
        description="Excellent, Good, Average, Poor etc."
    )
    evaluation_summary: str = Field(
        description="Short summary of the answer evaluation."
    )
    recommendation_for_agent5: str = Field(
        description="""
        Recommendation for the Interview Manager.
        Examples:
        - Move to next question.
        - Ask the same question with a hint.
        - Increase difficulty level.
        - Skip current topic.
        - Terminate interview.
        - Move to next interview step.
        """
    )


# Main Function

def evaluate_answer(
    question,
    answer,
    resume=None,
    jd=None,
    matcher=None,
    interview_plan=None,
    voice_metrics=None,
):
    """
    Evaluate a single candidate answer.

    FIX (critical bug): previously this function silently discarded the real
    `question`/`answer` arguments and evaluated a hardcoded dummy Q&A pair on
    every call. It now uses the real arguments that are passed in.

    resume/jd/matcher/interview_plan are optional: if the caller already has
    these in memory (as main.py does), they are used directly. They are only
    re-read from disk as a fallback, e.g. for standalone testing via
    `python answer_evaluator.py`.

    voice_metrics (optional): {"duration_seconds", "words_per_minute",
    "no_response"} from voice_io.listen_for_answer(), used ONLY as
    auxiliary context to help judge confidence — e.g. a very long pause
    before a short answer, or a rushed/fragmented delivery. This does not
    replace judging the actual content of the answer.
    """
    load_dotenv()

    if resume is None:
        with open("data/resume.json", "r", encoding="utf-8") as f:
            resume = json.load(f)
    if jd is None:
        with open("data/jd.json", "r", encoding="utf-8") as f:
            jd = json.load(f)
    if matcher is None:
        with open("data/matcher.json", "r", encoding="utf-8") as f:
            matcher = json.load(f)
    if interview_plan is None:
        with open("data/planner.json", "r", encoding="utf-8") as f:
            interview_plan = json.load(f)

    # Cheap deterministic short-circuit: don't waste an LLM call on an
    # empty/whitespace-only answer.
    if not answer or not answer.strip():
        return EvaluationReport(
            relevant_answer=False,
            technically_correct=False,
            logically_correct=False,
            offensive_language=False,
            need_warning=False,
            need_hint=True,
            retry_question=True,
            candidate_understands_topic=False,
            confidence_score=0,
            answer_quality="Poor",
            evaluation_summary="No answer was provided.",
            recommendation_for_agent5="Retry the question and offer a hint.",
        )

    if voice_metrics:
        if voice_metrics.get("no_response"):
            voice_notes = "The candidate did not respond within the allotted time (silence)."
        else:
            voice_notes = (
                f"Response took about {voice_metrics.get('duration_seconds', 0)} seconds, "
                f"spoken at roughly {voice_metrics.get('words_per_minute', 0)} words per minute."
            )
    else:
        voice_notes = "N/A (text-based interview, no voice data)."

    evaluator_prompt = PromptTemplate(
        template="""
You are a Senior Technical Interview Answer Evaluator AI.
Your task is ONLY to evaluate the candidate's answer.

INPUTS:
--------------------------------------------------
Structured Resume:
{resume}

--------------------------------------------------
Structured Job Description:
{jd}

--------------------------------------------------
Match Result:
{matcher}

--------------------------------------------------
Interview Plan:
{interview_plan}

--------------------------------------------------
Current Interview Question:
{question}

--------------------------------------------------
Candidate Answer:
{answer}

--------------------------------------------------
Voice Delivery Notes (auxiliary context only — the candidate's spoken
answer was transcribed to text above; these notes describe HOW it was
delivered, not what was said. Use them only as a minor supporting signal
for confidence_score, never to override your judgment of the actual
content):
{voice_notes}

--------------------------------------------------

YOUR RESPONSIBILITIES:
1. Evaluate the candidate's answer.
2. Check whether the answer is relevant.
3. Check whether the answer is technically correct.
4. Check whether the answer is logically correct.
5. Detect offensive or inappropriate language.
6. Determine whether the candidate understands the topic.
7. Determine whether a hint is required.
8. Determine whether the same question should be asked again.
9. Generate a confidence score between 0 and 100.
10. Generate a recommendation for Agent-5.

--------------------------------------------------

RULES:
1. NEVER conduct the interview.
2. NEVER ask interview questions.
3. NEVER compare the resume and JD unless required for context.
4. NEVER generate the next question.
5. NEVER manage interview flow.
6. NEVER maintain interview state.
7. ONLY evaluate the provided answer.
8. If the answer is completely unrelated, mark it as not relevant.
9. If the answer is partially correct, suggest providing a hint.
10. If offensive language is detected, recommend a warning.
11. Be strict but fair in evaluation.

--------------------------------------------------

EXAMPLES:

If answer is correct:
Recommendation:
Move to the next question.

-----------------------------------------

If answer is partially correct:
Recommendation:
Ask the same question again with a hint.

-----------------------------------------

If answer is poor:
Recommendation:
Retry the question.

-----------------------------------------

If answer is offensive:
Recommendation:
Give a warning message.

-----------------------------------------
Return the output STRICTLY according to the provided schema.
""",
        input_variables=[
            "resume",
            "jd",
            "matcher",
            "interview_plan",
            "question",
            "answer",
            "voice_notes",
        ],
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile")

    structured_llm = llm.with_structured_output(EvaluationReport)

    chain = evaluator_prompt | structured_llm

    evaluation_result = chain.invoke(
        {
            "resume": json.dumps(resume, indent=2),
            "jd": json.dumps(jd, indent=2),
            "matcher": json.dumps(matcher, indent=2),
            "interview_plan": json.dumps(interview_plan, indent=2),
            # FIX: use the real question/answer, not hardcoded dummy values.
            "question": question,
            "answer": answer,
            "voice_notes": voice_notes,
        }
    )
    return evaluation_result


# Standalone test
if __name__ == "__main__":
    print("Answer Evaluation Report (standalone test)...\n")
    result = evaluate_answer(
        question="What is a variable in Python?",
        answer="A variable is used to store values in Python.",
    )
    print(result.model_dump_json(indent=2))