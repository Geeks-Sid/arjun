"""Prompt templates and conversation formatting for VLM instruction tuning.

A :class:`PromptTemplate` carries a ``user_template`` with named ``str.format``
placeholders (e.g. ``{question}``, ``{findings}``) and an optional ``system``
string. :func:`format_conversation` renders fields into the template and
builds the :class:`~medfm.core.sample.ConversationTurn` list consumed by
:func:`medfm.data.textprep.tokenize.build_supervised_example`.

Placeholder discipline: the template may only reference keys the caller
supplies — unknown placeholder keys raise :class:`TextPreprocessError` rather
than failing with a bare ``KeyError`` deep in formatting. Extra supplied
fields are ignored.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

from medfm.core.sample import ConversationTurn
from medfm.data.errors import TextPreprocessError

_FORMATTER = string.Formatter()


def _placeholder_names(template: str) -> set[str]:
    """Return the root placeholder names referenced by ``template``."""
    names: set[str] = set()
    for _, field_name, _, _ in _FORMATTER.parse(template):
        if field_name is not None:
            names.add(field_name.split(".")[0].split("[")[0])
    return names


def _render(template: str, fields: dict[str, str], template_id: str) -> str:
    """Render one template string, rejecting unknown placeholders."""
    unknown = _placeholder_names(template) - set(fields)
    if unknown:
        raise TextPreprocessError(
            f"prompt template {template_id!r} references unknown placeholder(s): {sorted(unknown)}"
        )
    try:
        rendered = template.format(**fields)
    except (ValueError, IndexError, KeyError) as exc:
        raise TextPreprocessError(f"prompt template {template_id!r} is malformed: {exc}") from exc
    if not rendered.strip():
        raise TextPreprocessError(f"prompt template {template_id!r} rendered to empty text")
    return rendered


@dataclass(frozen=True)
class PromptTemplate:
    """A versioned prompt template for one VLM task variant."""

    template_id: str
    user_template: str
    system: str | None = None
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.template_id:
            raise TextPreprocessError("PromptTemplate.template_id must be non-empty")
        if not self.user_template:
            raise TextPreprocessError("PromptTemplate.user_template must be non-empty")
        if not self.version:
            raise TextPreprocessError("PromptTemplate.version must be non-empty")
        if self.system is not None and not self.system.strip():
            raise TextPreprocessError("PromptTemplate.system, when set, must be non-blank")


class PromptTemplateRegistry:
    """In-memory registry of prompt templates; duplicate ids are rejected."""

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}

    def register(self, template: PromptTemplate) -> None:
        """Add ``template``; raise :class:`TextPreprocessError` on a duplicate id."""
        if template.template_id in self._templates:
            raise TextPreprocessError(f"prompt template id {template.template_id!r} is already registered")
        self._templates[template.template_id] = template

    def get(self, template_id: str) -> PromptTemplate:
        """Return the template for ``template_id``; raise if unknown."""
        try:
            return self._templates[template_id]
        except KeyError:
            raise TextPreprocessError(f"unknown prompt template id {template_id!r}") from None


def format_conversation(
    template: PromptTemplate,
    fields: dict[str, str],
    assistant_text: str | None = None,
) -> list[ConversationTurn]:
    """Render ``template`` with ``fields`` into a conversation turn list.

    Order: optional system turn, the rendered user turn, and — when
    ``assistant_text`` is given — the assistant reference turn (supervised
    content during training).
    """
    turns: list[ConversationTurn] = []
    if template.system is not None:
        turns.append(ConversationTurn(role="system", content=_render(template.system, fields, template.template_id)))
    turns.append(ConversationTurn(role="user", content=_render(template.user_template, fields, template.template_id)))
    if assistant_text is not None:
        turns.append(ConversationTurn(role="assistant", content=assistant_text))
    return turns
