"""Request classification, tool orchestration and evidence synthesis.

The design rule is that the orchestrator may only claim what actually ran:

* Classification and planning are deterministic rules, so the same input always
  produces the same plan and the behaviour is auditable.
* Every tool call is permission-checked, time-boxed and validated. Tools that
  change state (code execution, repository writes) are refused unless the caller
  passes an explicit approval *and* the deployment can isolate execution.
* The final answer is assembled from validated tool output and labelled with the
  tool that produced it. When a tool fails, the answer says so. When no tool
  applies and no model is available, the answer says that too.
* Retrieved document text is treated as untrusted evidence, quoted with a
  citation and never executed or followed as instructions.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from helix import numerics

READ_ONLY = 'read-only'
APPROVAL_REQUIRED = 'approval-required'

MAX_MESSAGE_CHARS = 4000
TOOL_TIMEOUT_S = 5.0

_MATH_HINTS = re.compile(
    r"(?i)\b(differentiat\w*|integr\w*|derivative|solve|simplif\w*|limit|equation|"
    r"algebra|symbolic|x\s*\*{2}\s*\d|\\frac)\b")
_FORECAST_HINTS = re.compile(r"(?i)\b(forecast|predict|trend|projection|time series|extrapolat\w*)\b")
_SEARCH_HINTS = re.compile(r"(?i)\b(find|search|source|sources|cite|citation|according to|document|documents|index|evidence)\b")
_CODE_HINTS = re.compile(r"(?i)\b(run|execute|traceback|exception|script|function|python|debug|refactor)\b")


@dataclass
class ToolCall:
    tool: str
    permission: str
    ok: bool
    duration_ms: float
    summary: str = ''
    error: Optional[str] = None
    input_sha256: str = ''

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    classification: str
    steps: List[str]
    tools: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentResult:
    request_id: str
    message_sha256: str
    classification: str
    plan: List[str]
    tool_calls: List[ToolCall] = field(default_factory=list)
    answer: str = ''
    evidence: List[dict] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    needs_model: bool = False
    complied: List[str] = field(default_factory=list)
    refused: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload['tool_calls'] = [call.to_dict() for call in self.tool_calls]
        return payload


def classify(message: str) -> str:
    """Deterministic intent classification used to choose a plan."""
    text = (message or '').strip()
    if not text:
        return 'empty'
    if _MATH_HINTS.search(text):
        return 'math'
    if _FORECAST_HINTS.search(text):
        return 'forecast'
    if _SEARCH_HINTS.search(text):
        return 'search'
    if _CODE_HINTS.search(text):
        return 'code'
    return 'conversation'


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


class Orchestrator:
    """A small, auditable tool router.

    ``tools`` maps a tool name to a callable taking a dict of arguments and
    returning a JSON-serialisable dict. ``generate`` is an optional model
    callback used only for the conversation class.
    """

    PERMISSIONS = {
        'symbolic_math': READ_ONLY,
        'linear_forecast': READ_ONLY,
        'document_search': READ_ONLY,
        'code_execution': APPROVAL_REQUIRED,
    }

    def __init__(self, tools: Dict[str, Callable[[dict], dict]],
                 generate: Optional[Callable[[str], str]] = None,
                 documents_available: bool = False,
                 sandbox_available: bool = False):
        unknown = set(tools) - set(self.PERMISSIONS)
        if unknown:
            raise ValueError(f'tools without a declared permission: {sorted(unknown)}')
        self.tools = tools
        self.generate = generate
        self.documents_available = documents_available
        self.sandbox_available = sandbox_available
        self._last_call: Dict[str, float] = {}

    # -- planning ----------------------------------------------------------
    def plan(self, message: str) -> Plan:
        classification = classify(message)
        if classification == 'math':
            return Plan(classification, ['extract expression', 'run SymPy', 'report result with tool attribution'],
                        ['symbolic_math'])
        if classification == 'forecast':
            return Plan(classification, ['parse numeric series', 'fit and score trend', 'report forecast with interval'],
                        ['linear_forecast'])
        if classification == 'search':
            tools = ['document_search'] if self.documents_available else []
            steps = ['search local document index', 'cite the retrieved passages'] if tools else \
                    ['no local document index is configured', 'state the limitation instead of searching']
            return Plan(classification, steps, tools)
        if classification == 'code':
            steps = ['static analysis only',
                     'execution requires approval and a working sandbox']
            return Plan(classification, steps, ['code_execution'])
        return Plan(classification, ['answer with the language model'], [])

    # -- execution ---------------------------------------------------------
    def _call(self, result: AgentResult, tool: str, arguments: dict,
              approved: bool, input_text: str) -> Optional[dict]:
        permission = self.PERMISSIONS.get(tool, APPROVAL_REQUIRED)
        if permission == APPROVAL_REQUIRED and not approved:
            result.refused.append(f'{tool} was not run: it needs explicit approval.')
            return None
        if tool == 'code_execution' and not self.sandbox_available:
            result.refused.append('code_execution was not run: no isolation backend is available.')
            return None
        handler = self.tools.get(tool)
        if handler is None:
            result.limitations.append(f'{tool} is not registered on this deployment.')
            return None
        started = time.perf_counter()
        try:
            output = handler(arguments)
            duration = (time.perf_counter() - started) * 1000.0
            if duration > TOOL_TIMEOUT_S * 1000:
                result.tool_calls.append(ToolCall(tool, permission, False, duration,
                                                  error='tool exceeded its time budget',
                                                  input_sha256=_digest(input_text)))
                result.limitations.append(f'{tool} exceeded its time budget and its result was discarded.')
                return None
            if not isinstance(output, dict):
                raise TypeError('tool returned a non-dict result')
            result.tool_calls.append(ToolCall(tool, permission, True, duration,
                                              summary=str(output.get('summary', ''))[:200],
                                              input_sha256=_digest(input_text)))
            result.complied.append(tool)
            return output
        except Exception as exc:  # noqa: BLE001 - tool failures are data, not crashes
            duration = (time.perf_counter() - started) * 1000.0
            message = f'{type(exc).__name__}: {exc}'
            result.tool_calls.append(ToolCall(tool, permission, False, duration,
                                              error=message[:200], input_sha256=_digest(input_text)))
            result.limitations.append(f'{tool} failed and produced no usable result.')
            return None

    def run(self, message: str, request_id: str, approved: bool = False,
            series: Optional[Sequence[float]] = None, horizon: int = 3,
            expression: str = '', operation: str = 'differentiate',
            variable: str = 'x') -> AgentResult:
        text = (message or '').strip()
        if len(text) > MAX_MESSAGE_CHARS:
            raise ValueError(f'message must be at most {MAX_MESSAGE_CHARS} characters')
        result = AgentResult(request_id=request_id, message_sha256=_digest(text),
                             classification=classify(text), plan=[])
        plan = self.plan(text)
        result.plan = plan.steps
        result.classification = plan.classification

        if plan.classification == 'empty':
            result.answer = 'Send a question, a calculation or a document query.'
            return result

        if plan.classification == 'math':
            output = self._call(result, 'symbolic_math',
                                {'operation': operation, 'expression': expression or text,
                                 'variable': variable}, True, expression or text)
            if output and output.get('ok'):
                steps = '\n'.join(output.get('steps', []))
                verifier = output.get('verified_by') or 'sympy'
                version = output.get('sympy_version') or ''
                result.answer = (f'{output.get("result")}\n\n{steps}\n\n'
                                 f'Computed by {verifier}{" " + version if version else ""} '
                                 f'(symbolic tool output, not model output).')
            elif output:
                result.answer = f'SymPy could not evaluate that expression: {output.get("error", "unknown error")}.'
            else:
                result.answer = 'The symbolic tool was unavailable for this request.'

        elif plan.classification == 'forecast':
            if not series:
                result.answer = ('Forecasting needs at least three numeric points. Send them as '
                                 '`series` (a list of numbers) or as a CSV column.')
                result.limitations.append('no numeric series was supplied')
            else:
                output = self._call(result, 'linear_forecast',
                                    {'series': series, 'horizon': horizon}, True, str(len(series)))
                if output:
                    first = output['predictions'][0]
                    result.answer = (
                        f"Least-squares trend forecast: {first['yhat']:.4g} at step {first['x']:.0f}, "
                        f"{int(output['interval'] * 100)}% prediction interval "
                        f"[{first['lower']:.4g}, {first['upper']:.4g}]. "
                        f"Hold-out MAE {output['holdout']['mae']:.4g} vs last-value baseline "
                        f"{output['holdout']['baseline_mae_last_value']:.4g} "
                        f"({'beats' if output['holdout']['beats_baseline'] else 'does not beat'} the baseline). "
                        f"Computed by helix.numerics ({output['backend']} backend).")
                    result.limitations.extend(output['assumptions'])
                else:
                    result.answer = 'The numeric forecast tool did not produce a usable result.'

        elif plan.classification == 'search':
            if not self.documents_available:
                result.answer = ('No local document index is configured on this deployment, so nothing was '
                                 'searched and no source can be cited. Web search is not available.')
                result.limitations.append('retrieval not configured')
            else:
                output = self._call(result, 'document_search', {'query': text, 'k': 4}, True, text)
                if output and output.get('results'):
                    result.evidence = output['results']
                    lines = [f"[{index + 1}] {item['citation']}\n{item['snippet']}"
                             for index, item in enumerate(output['results'])]
                    result.answer = ('Retrieved from the local index (lexical + hashed similarity ranking, '
                                     'not a semantic model):\n\n' + '\n\n'.join(lines))
                elif output:
                    result.answer = 'The local index returned no passage for that query.'
                else:
                    result.answer = 'The document search tool failed; no evidence is available.'

        elif plan.classification == 'code':
            output = self._call(result, 'code_execution', {'code': text}, approved, text)
            if output and output.get('ok'):
                result.answer = (f"Sandbox output ({output.get('isolation', 'unknown isolation')}):\n"
                                 f"{output.get('output', '').strip() or '(no output)'}")
            elif output:
                result.answer = f"Execution finished with an error:\n{output.get('error', '')}"
            else:
                result.answer = ('I can analyze code without approval, but running it needs the approval '
                                 'flag and a working isolation backend. Nothing was executed.')

        else:  # conversation
            result.needs_model = True
            if self.generate is None:
                result.answer = ('This deployment has no language model loaded, so no generated answer is '
                                 'available. Tool-backed answers (math, forecasting, document search) still work.')
                result.limitations.append('language model unavailable')
            else:
                result.answer = self.generate(text)
                result.complied.append('language_model')

        result.limitations.append('The orchestrator only reports tools it actually ran.')
        return result
