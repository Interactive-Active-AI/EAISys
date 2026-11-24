"""Persistent versioned operational Playbook & zero-privilege Memory Governor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


def default_playbooks_dir() -> Path:
    pkg_pb = Path(__file__).resolve().parents[1] / "_bundled" / "configs" / "playbooks"
    if pkg_pb.is_dir():
        return pkg_pb
    return Path(__file__).resolve().parents[3] / "configs" / "playbooks"


@dataclass(frozen=True)
class PlaybookSection:
    tag: str
    title: str
    rules: tuple[str, ...]


class PlaybookStore:
    """Loads versioned zero-privilege Playbook policies (inspired by RobotUse & Skill2Real Governor)."""

    def __init__(self, version: str, sections: list[PlaybookSection]) -> None:
        self.version = version
        self.sections = sections

    @classmethod
    def load(cls, version: str = "v1", playbooks_dir: str | Path | None = None) -> "PlaybookStore":
        pdir = Path(playbooks_dir) if playbooks_dir else default_playbooks_dir()
        path = pdir / f"{version}.md"
        if not path.is_file():
            return cls(version=version, sections=[])
        return cls.parse_markdown(path.read_text(encoding="utf-8"), version=version)

    @classmethod
    def parse_markdown(cls, text: str, version: str = "v1") -> "PlaybookStore":
        sections: list[PlaybookSection] = []
        cur_tag = "universal"
        cur_title = "General"
        cur_rules: list[str] = []

        def _flush() -> None:
            if cur_rules:
                sections.append(PlaybookSection(tag=cur_tag, title=cur_title, rules=tuple(cur_rules)))

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("## "):
                _flush()
                cur_rules = []
                m = re.match(r"^##\s+\[([a-z0-9_-]+)\]\s*(.*)$", line, flags=re.I)
                if m:
                    cur_tag = m.group(1).lower()
                    cur_title = m.group(2).strip() or cur_tag
                else:
                    cur_tag = "universal"
                    cur_title = line[3:].strip()
            elif line.startswith("- "):
                cur_rules.append(line[2:].strip())
        _flush()
        return cls(version=version, sections=sections)

    def select_rules(
        self,
        task_instruction: str,
        *,
        is_bimanual: bool = False,
        max_rules: int = 6,
    ) -> list[str]:
        """Select compact, context-relevant zero-privilege playbook rules."""
        low = task_instruction.lower()
        ordered_tags: list[str] = []
        if is_bimanual or re.search(r"\b(two-arm|bimanual|both arms|pot|handover)\b", low):
            ordered_tags.append("bimanual")
        if re.search(r"\b(door|hinge|drawer|cabinet|latch)\b", low) or (
            not is_bimanual and re.search(r"\bhandle\b", low) and not re.search(r"\b(nut|ring|peg)\b", low)
        ):
            ordered_tags.append("articulated")
        if re.search(r"\b(bin|sort|organizer|carton|milk|can|bread|cereal|compartment)\b", low):
            ordered_tags.append("sorting")
        if re.search(r"\b(nut|peg|ring|insert|hole|assembly|hang)\b", low):
            ordered_tags.append("insertion")
        if re.search(r"\b(wipe|wiping|sweep|sponge|marker|dirt|stain)\b", low):
            ordered_tags.append("wiping")
        ordered_tags.append("universal")

        sec_by_tag = {sec.tag: sec for sec in self.sections}
        chosen: list[str] = []
        for tag in ordered_tags:
            sec = sec_by_tag.get(tag)
            if sec is not None:
                chosen.extend(sec.rules)
        return chosen[:max_rules]

    def append_governed_lesson(
        self,
        tag: str,
        lesson: str,
        output_path: str | Path,
    ) -> bool:
        """Skill2Real-style Memory Governor: reject any lesson containing privileged metrics before saving."""
        clean = lesson.strip().lstrip("- ").strip()
        if not clean:
            return False
        # Reject privileged simulator terms or exact cm/coordinate cheats.
        if re.search(r"\b\d+(?:\.\d+)?\s*(?:cm|mm)\b", clean, flags=re.I):
            return False
        forbidden = ("cubea_pos", "cubeb_pos", "table_z", "table_offset", "ground_truth", "oracle", "holding_object")
        if any(tok in clean.lower() for tok in forbidden):
            return False

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        existing = out_p.read_text(encoding="utf-8") if out_p.is_file() else f"# EAISys Playbook ({self.version})\n"
        header = f"## [{tag.lower()}]"
        if clean in existing:
            return False
        if header in existing:
            idx = existing.index(header)
            nl = existing.find("\n", idx)
            insert_at = len(existing) if nl == -1 else nl + 1
            updated = existing[:insert_at] + f"- {clean}\n" + existing[insert_at:]
        else:
            updated = existing.rstrip() + f"\n\n{header} Learned Operational Lessons\n- {clean}\n"
        out_p.write_text(updated, encoding="utf-8")
        return True
