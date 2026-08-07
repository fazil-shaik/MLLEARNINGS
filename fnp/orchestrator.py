from agents import PERSONAS, MODERATOR_SYSTEM, call_agent


def format_transcript(topic: str, transcript: list[dict]) -> str:
    """Turns the shared history into plain text every agent can read."""
    lines = [f"TOPIC UNDER DEBATE: {topic}", ""]
    for turn in transcript:
        lines.append(f"[{turn['speaker']}]: {turn['text']}")
    lines.append("")
    lines.append("Continue the debate. Respond as your persona, in character.")
    return "\n".join(lines)


def run_roundtable(topic: str, rounds: int = 2, verbose: bool = True) -> dict:
    """
    Runs the full debate and returns a dict with the transcript and the
    moderator's final synthesis.
    """
    transcript: list[dict] = []

    for round_num in range(1, rounds + 1):
        if verbose:
            print(f"\n{'=' * 60}\nROUND {round_num}\n{'=' * 60}")

        for name, persona in PERSONAS.items():
            conversation_text = format_transcript(topic, transcript)
            reply = call_agent(persona["system"], conversation_text)
            transcript.append({"speaker": name, "text": reply})

            if verbose:
                print(f"\n--- {name} ---\n{reply}")

    # Moderator reads the full transcript and closes the debate.
    if verbose:
        print(f"\n{'=' * 60}\nMODERATOR'S SYNTHESIS\n{'=' * 60}")

    moderator_input = format_transcript(topic, transcript)
    moderator_input = moderator_input.replace(
        "Continue the debate. Respond as your persona, in character.",
        "Write your closing synthesis now.",
    )
    verdict = call_agent(MODERATOR_SYSTEM, moderator_input)

    if verbose:
        print(f"\n{verdict}")

    return {"topic": topic, "transcript": transcript, "verdict": verdict}