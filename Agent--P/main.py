import json
import os

from interview_controller import InterviewController
from interview_manager import generate_bot_message
from answer_evaluator import evaluate_answer
from agent7_report import generate_final_report

# Set to True to run the interview by voice (pyttsx3 + Google speech
# recognition via the `voice_io` module) instead of typed input/output.
# Falls back to text automatically if voice_io can't be imported (e.g.
# pyaudio isn't installed on this machine).
VOICE_MODE = True

voice_io = None
if VOICE_MODE:
    try:
        import voice_io
    except Exception as e:
        print(f"\n[voice_io unavailable ({e}) — falling back to text mode]\n")
        voice_io = None


def ask_and_capture(bot_text: str):
    """
    Speaks/prints the bot's message and, if it expects an answer, captures
    the candidate's response (by voice or by typing).

    Returns (answer_text, voice_metrics_or_None).
    """
    if voice_io:
        voice_io.speak(bot_text)
        print("\nBOT :", bot_text)
        print("(listening... you have up to ~6-7 seconds of silence before the bot moves on)")
        answer, voice_metrics = voice_io.listen_for_answer(pause_seconds=6.5, timeout=8)
        if voice_metrics.get("no_response"):
            print("YOU : (no response detected)")
        else:
            print("YOU :", answer)
        return answer, voice_metrics
    else:
        print("\nBOT :", bot_text)
        answer = input("\nYOU : ")
        return answer, None


def main():
    with open("data/resume.json", "r", encoding="utf-8") as f:
        resume = json.load(f)
    with open("data/jd.json", "r", encoding="utf-8") as f:
        jd = json.load(f)
    with open("data/matcher.json", "r", encoding="utf-8") as f:
        matcher = json.load(f)
    with open("data/planner.json", "r", encoding="utf-8") as f:
        interview_plan = json.load(f)

    if not interview_plan["interview_eligible"]:
        print("\nCandidate is not eligible.")
        print(interview_plan["reason"])
        return

    print("\nInterview Started...\n")

    controller = InterviewController(interview_plan)
    conversation_history = []   # simple {question, answer} pairs, used for phrasing context
    interview_log = []          # rich per-turn log with step/skill/tier + evaluation, for Agent 7
    last_evaluation = None
    was_terminated = False
    termination_reason = None

    while True:
        directive = controller.next_action(last_evaluation)

        output = generate_bot_message(
            directive=directive,
            resume=resume,
            jd=jd,
            matcher=matcher,
            interview_plan=interview_plan,
            conversation_history=conversation_history,
            last_evaluation=last_evaluation,
        )

        if directive["terminate"] or directive["complete"] or not output.asked_question:
            # Final message - just deliver it, no answer expected.
            print("\nBOT :", output.bot_response)
            
            if voice_io:
                voice_io.speak(output.bot_response)
            was_terminated = bool(directive["terminate"])
            termination_reason = directive.get("termination_reason")
            break

        answer, voice_metrics = ask_and_capture(output.bot_response)

        conversation_history.append({"question": output.bot_response, "answer": answer})

        last_evaluation = evaluate_answer(
            question=output.bot_response,
            answer=answer,
            resume=resume,
            jd=jd,
            matcher=matcher,
            interview_plan=interview_plan,
            voice_metrics=voice_metrics,
        )
        print("\nEvaluation Completed.")

        interview_log.append({
            "step_number": directive["step_number"],
            "step_name": directive["step_name"],
            "skill_name": directive.get("skill_name"),
            "tier": directive.get("tier"),
            "topic": directive.get("topic"),
            "question": output.bot_response,
            "answer": answer,
            "voice_metrics": voice_metrics,
            "evaluation": last_evaluation.model_dump(),
        })

    print("\nInterview Finished.\n")

    os.makedirs("data", exist_ok=True)
    with open("data/interview_log.json", "w", encoding="utf-8") as f:
        json.dump(interview_log, f, indent=2)

    if not interview_log:
        print("No questions were answered — skipping final report.")
        return

    print("Generating final interview report (Agent 7)...")
    final_report = generate_final_report(
        interview_log=interview_log,
        matcher=matcher,
        resume=resume,
        jd=jd,
        interview_plan=interview_plan,
        was_terminated=was_terminated,
        termination_reason=termination_reason,
    )

    with open("data/final_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\n----- FINAL REPORT (data/final_report.json) -----")
    print(f"Candidate: {final_report['candidate_name']}")
    print(f"Status: {final_report['interview_status']}")
    print(f"Requirement Match Score: {final_report['requirement_match_score']}")
    print(f"Interview Performance Score: {final_report['interview_performance_score']}")
    print(f"Combined Score: {final_report['combined_score']}")
    print(f"Recommendation: {final_report['overall_recommendation']}")
    print(f"\nHR Summary:\n{final_report['hr_summary']}")


if __name__ == "__main__":
    main()
