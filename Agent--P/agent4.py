from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import os


class SkillPlan(BaseModel):
    skill_name: str
    mandatory: bool
    basic_questions: int
    intermediate_questions: int
    advanced_questions: int
    terminate_if_failed: bool


class StepPlan(BaseModel):
    step_number: int
    step_name: str
    estimated_questions: int
    topics: List[str]


class InterviewPlan(BaseModel):
    interview_eligible: bool
    interview_status: str
    reason: str
    total_steps: int
    total_estimated_questions: int
    step_plan: List[StepPlan]
    programming_languages: List[SkillPlan]
    libraries_frameworks: List[SkillPlan]
    tools: List[SkillPlan]
    project_topics: List[str]
    reasoning_topics: List[str]


EXPECTED_STEP_NAMES = [
    "Personal Information",
    "Programming Languages",
    "Libraries / Frameworks",
    "Tools",
    "Projects",
    "Reasoning",
]


def parse_interview_plan(
    resume_data: Optional[dict] = None,
    jd_data: Optional[dict] = None,
    matcher_data: Optional[dict] = None,
    resume_path: str = "data/resume.json",
    jd_path: str = "data/jd.json",
    matcher_path: str = "data/matcher.json",
) -> InterviewPlan:
    """
    FIXES vs. the original:
    1. Now accepts resume/jd/matcher data directly as dicts (falls back to
       disk if not given), so the Streamlit pipeline can pass results
       straight from Agents 1-3 without a disk round-trip.
    2. Soft validation: warns (does not crash) if the plan's steps don't
       match your required six-step order — useful for catching LLM drift
       during testing without breaking the pipeline in production.
    """
    load_dotenv()

    if resume_data is None:
        with open(resume_path, "r", encoding="utf-8") as f:
            resume_data = json.load(f)
    if jd_data is None:
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_data = json.load(f)
    if matcher_data is None:
        with open(matcher_path, "r", encoding="utf-8") as f:
            matcher_data = json.load(f)

    planner_prompt = PromptTemplate(
        template="""
You are a Senior Technical Interview Planner AI.
Your task is NOT to conduct the interview.
Your task is NOT to generate actual interview questions.
Your task is ONLY to create an interview plan.

INPUTS:
1. Structured Resume
{resume}
2. Structured Job Description
{jd}
3. Match Result
{matcher}

Rules:
1. If the candidate is NOT eligible for interview, return interview_eligible=False and explain the reason.
2. If the candidate IS eligible, create a structured interview plan.
3. The interview MUST follow exactly six steps:
- Personal Information
- Programming Languages
- Libraries / Frameworks
- Tools
- Projects
- Reasoning

4. Include ONLY the relevant skills according to:
- Resume
- Job Description
- Match Result

5. Required skills should be marked as mandatory.
6. Mandatory skills MAY terminate the interview if the candidate completely fails them.
7. Do NOT include unrelated skills.
8. Estimate how many Basic, Intermediate and Advanced questions should be asked for each skill.
9. Estimate the total number of questions for the entire interview.
10. Do NOT generate actual interview questions.
11. Return the output STRICTLY according to the provided schema.
""",
        input_variables=["resume", "jd", "matcher"],
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile")
    structured_llm = llm.with_structured_output(InterviewPlan)
    chain = planner_prompt | structured_llm

    try:
        structured_interview_plan = chain.invoke(
            {
                "resume": json.dumps(resume_data, indent=2),
                "jd": json.dumps(jd_data, indent=2),
                "matcher": json.dumps(matcher_data, indent=2),
            }
        )
    except Exception as e:
        raise RuntimeError(f"The AI model failed to generate the interview plan: {e}")

    actual_steps = [sp.step_name for sp in structured_interview_plan.step_plan]
    if structured_interview_plan.interview_eligible and actual_steps != EXPECTED_STEP_NAMES:
        print(f"[warning] Interview plan step order deviates from the required six steps: {actual_steps}")

    return structured_interview_plan


if __name__ == "__main__":
    print("Interview Planner Output...\n")
    result = parse_interview_plan()
    print(result.model_dump_json(indent=2))

    os.makedirs("data", exist_ok=True)
    with open("data/planner.json", "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))
    print("\nInterview plan successfully saved in data/planner.json")