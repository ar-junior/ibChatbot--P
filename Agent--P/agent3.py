from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import json


class MatchResult(BaseModel):
    requirement_match_score: float = Field(
        description="Overall requirement match score between 0 and 100."
    )
    matched_skills: List[str] = Field(default_factory=list, description="Skills present in both the resume and the job description.")
    missing_required_skills: List[str] = Field(default_factory=list, description="Required skills from the job description that are missing in the resume.")
    matched_preferred_skills: List[str] = Field(default_factory=list, description="Preferred skills that are also present in the resume.")
    strengths: List[str] = Field(default_factory=list, description="Major strengths of the candidate for this job.")
    weaknesses: List[str] = Field(default_factory=list, description="Major weaknesses or gaps for this job.")
    interview_eligible: bool = Field(description="Whether the candidate should proceed to the AI interview.")
    recommendation: str = Field(description="Short recommendation such as Strong Match, Moderate Match, or Weak Match.")
    reason: str = Field(description="Brief explanation supporting the recommendation.")


def parse_matcher(
    resume_data: Optional[dict] = None,
    jd_data: Optional[dict] = None,
    resume_path: str = "data/resume.json",
    jd_path: str = "data/jd.json",
    eligibility_floor: float = 30.0,
) -> MatchResult:
    """
    Compares a structured resume against a structured JD and produces a
    MatchResult.

    FIXES vs. the original:
    1. Now accepts `resume_data`/`jd_data` directly (as dicts) so the
       Streamlit app can pass in-memory results from Agent 1/2 without a
       disk round-trip. Falls back to reading from `resume_path`/`jd_path`
       if not provided, so the old file-based workflow/standalone testing
       still works unchanged.
    2. Added a deterministic safety net: even if the LLM says
       interview_eligible=True, if its own computed requirement_match_score
       is below `eligibility_floor` (default 30), eligibility is forced to
       False. This guards against inconsistent LLM judgments near the
       borderline — it does not override a well-reasoned "eligible" call
       at any real match score, only catches contradictions.
    3. Added error handling for missing files / LLM failures.
    """
    load_dotenv()

    if resume_data is None:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume JSON not found at: {resume_path}")
        with open(resume_path, "r", encoding="utf-8") as f:
            resume_data = json.load(f)

    if jd_data is None:
        if not os.path.exists(jd_path):
            raise FileNotFoundError(f"JD JSON not found at: {jd_path}")
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_data = json.load(f)

    matcher_prompt = PromptTemplate(
        template="""
    You are an experienced technical recruiter.
    Your task is to compare the candidate's structured resume with the structured job description.

    Rules:
    - Compare only the provided information.
    - Do not assume or invent information.
    - Match skills semantically when appropriate (for example, Git may satisfy part of "Git and GitHub", but do not over-match unrelated skills).
    - Distinguish technical skills from soft skills.
    - Use projects, education and experience as supporting evidence.
    - Base the requirement_match_score primarily on required skills.
    - Return the output exactly according to the provided schema.

    Structured Resume
    {resume}
    Structured Job Description:
    {job_description}
    """,
        input_variables=["resume", "job_description"],
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile")
    structured_llm = llm.with_structured_output(MatchResult)
    chain = matcher_prompt | structured_llm

    try:
        structured_matcher = chain.invoke(
            {
                "job_description": json.dumps(jd_data, indent=2),
                "resume": json.dumps(resume_data, indent=2),
            }
        )
    except Exception as e:
        raise RuntimeError(f"The AI model failed to compute the requirement match: {e}")

    # Deterministic safety net (see docstring point 2).
    if structured_matcher.requirement_match_score < eligibility_floor:
        structured_matcher.interview_eligible = False

    return structured_matcher


if __name__ == "__main__":
    print("structured matcher...")
    result = parse_matcher()
    print(result.model_dump_json(indent=2))

    os.makedirs("data", exist_ok=True)
    with open("data/matcher.json", "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))
    print("\nMatch result successfully saved in data/matcher.json")