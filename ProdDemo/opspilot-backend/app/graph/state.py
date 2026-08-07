from typing import TypedDict, Literal, Optional


class OpsPilotState(TypedDict, total=False):
    thread_id: str
    event_type: Literal["lead", "ticket", "ops_task", "unknown"]
    payload: dict            # raw incoming event data
    department: Optional[str]  # sales | support | ops, set by classifier
    crew_result: Optional[str]  # final text output from the crew that ran
    db_record_id: Optional[int]
    needs_human: bool
