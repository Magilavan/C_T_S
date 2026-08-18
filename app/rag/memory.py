"""Per-session conversation memory (locked-in decision: session-scoped,
resets on new conversation — see design doc section 6.1). In-memory dict
is fine for a hackathon demo; swap for Redis if this needs to survive
process restarts."""
from dataclasses import dataclass, field
import time
from app.core.config import settings


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


class SessionMemory:
    def __init__(self, max_turns: int = 8):
        self._sessions: dict[str, list[Turn]] = {}
        self._active_drug: dict[str, str | None] = {}
        self._states: dict[str, dict] = {}
        self.max_turns = max_turns

    def _key(self, session_id: str, user_id: int | None = None) -> str:
        """Create an isolated key scoped by user_id and session_id."""
        if user_id is not None:
            return f"u{user_id}:{session_id}"
        return session_id

    def get_history(self, session_id: str, user_id: int | None = None) -> list[Turn]:
        k = self._key(session_id, user_id)
        return self._sessions.get(k, [])

    def add_turn(self, session_id: str, role: str, content: str, user_id: int | None = None):
        k = self._key(session_id, user_id)
        self._sessions.setdefault(k, []).append(Turn(role, content))
        self._sessions[k] = self._sessions[k][-self.max_turns:]

    def get_active_drug(self, session_id: str, user_id: int | None = None) -> str | None:
        k = self._key(session_id, user_id)
        return self._active_drug.get(k)

    def set_active_drug(self, session_id: str, drug_name: str, user_id: int | None = None):
        k = self._key(session_id, user_id)
        self._active_drug[k] = drug_name
        self.update_state(session_id, {"drug": drug_name}, user_id=user_id)

    def get_state(self, session_id: str, user_id: int | None = None) -> dict:
        k = self._key(session_id, user_id)
        if k not in self._states:
            self._states[k] = {
                "drug": None,
                "active_document": None,
                "current_indication": None,
                "current_population": None,
                "current_topic": None,
                "current_section": None,
                "last_answer_entities": [],
                "last_question": None,
                "last_answer": None
            }
        return self._states[k]

    def update_state(self, session_id: str, updates: dict, user_id: int | None = None):
        k = self._key(session_id, user_id)
        state = self.get_state(session_id, user_id=user_id)
        for field_name, v in updates.items():
            state[field_name] = v
        # Synchronize active_drug if present
        if state.get("drug"):
            self._active_drug[k] = state["drug"]

    def format_history(self, session_id: str, user_id: int | None = None) -> str:
        """Format recent history compactly for LLM context.

        Only includes the last MAX_HISTORY_TURNS turns, and truncates
        long assistant answers to reduce token consumption.
        """
        turns = self.get_history(session_id, user_id=user_id)
        if not turns:
            return "(no prior turns)"

        # Only include the most recent turns based on config
        max_turns = getattr(settings, 'max_history_turns', 4)
        recent = turns[-max_turns:]

        lines = []
        for t in recent:
            content = t.content
            # Truncate long assistant answers in history
            if t.role == "assistant" and len(content) > 300:
                content = content[:300] + "..."
            lines.append(f"{t.role.upper()}: {content}")
        return "\n".join(lines)


memory = SessionMemory()

