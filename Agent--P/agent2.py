from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
import os


class JobDescription(BaseModel):
    job_title: str = Field(description="Job title")
    company_name: Optional[str] = Field(default=None, description="Company name")
    location: Optional[str] = Field(default=None, description="Job location")
    job_type: Optional[str] = Field(default=None, description="Employment type")
    experience_required: Optional[str] = Field(default=None, description="Required experience")
    education_requirements: List[str] = Field(default_factory=list, description="Required education")
    required_skills: List[str] = Field(default_factory=list, description="Mandatory technical skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Preferred technical skills")
    responsibilities: List[str] = Field(default_factory=list, description="Job responsibilities")
    nice_to_have: List[str] = Field(default_factory=list, description="Additional preferred qualifications")
    minimum_eligibility: List[str] = Field(default_factory=list, description="Minimum eligibility criteria")


def parse_jd(jd_path: str = "data/jd.txt") -> JobDescription:
    """
    Parses a plain-text Job Description file into a JobDescription.

    FIXES vs. the original:
    1. Standardized default path to `data/jd.txt` (the original mixed
       conventions: agent1 read from the project root, agent2 read from an
       `agent/` subfolder — now both agents' inputs live under `data/`).
    2. Added error handling for a missing/empty file and LLM failures.

    For now the JD is provided manually (you copy/paste or edit
    `data/jd.txt` yourself) rather than uploaded through the website, per
    your current requirement — this function just takes whatever path you
    give it, so wiring up an upload later is a one-line change in
    pipeline_utils.py, not a change here.
    """
    load_dotenv()

    if not os.path.exists(jd_path):
        raise FileNotFoundError(f"Job description file not found at: {jd_path}")

    try:
        loader = TextLoader(jd_path)
        documents = loader.load()
    except Exception as e:
        raise RuntimeError(f"Failed to read the JD file at {jd_path}: {e}")

    job_description = documents[0].page_content if documents else ""
    if not job_description.strip():
        raise ValueError(f"The JD file at {jd_path} appears to be empty.")

    jd_prompt = PromptTemplate(
        template="""
    You are an expert Job Description parser.

    Extract the information from the Job Description according to the provided schema.

    Rules:
    - Use only the information present in the Job Description.
    - Do not guess or invent any information.
    - If a field is missing, return null or an empty list.
    - Return information exactly according to the schema.

    Job Description:

    {job_description}
    """,
        input_variables=["job_description"],
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile")
    structured_llm = llm.with_structured_output(JobDescription)
    chain = jd_prompt | structured_llm

    try:
        structured_jd = chain.invoke({"job_description": job_description})
    except Exception as e:
        raise RuntimeError(f"The AI model failed to parse this job description: {e}")

    return structured_jd


if __name__ == "__main__":
    print("jd structured....")
    result = parse_jd()
    print(result.model_dump_json(indent=2))

    os.makedirs("data", exist_ok=True)
    with open("data/jd.json", "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))
    print("\nJD successfully saved in data/jd.json")