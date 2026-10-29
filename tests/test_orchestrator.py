"""Agent orchestration: classification, permissions, evidence and honesty."""
import json
from pathlib import Path

import pytest

import helix.web as web
from helix.orchestrator import Orchestrator, classify
from helix.retrieval import RetrievalIndex

FIXTURES = Path(__file__).parent / 'fixtures' / 'retrieval'


@pytest.mark.parametrize('message,expected', [
    ('Differentiate x**3 + 2*x', 'math'),
    ('what is the derivative of sin(x)?', 'math'),
    ('Forecast this trend for next quarter', 'forecast'),
    ('find the source about attention masking', 'search'),
    ('run this python script and show the traceback', 'code'),
    ('Write me a short poem about the sea', 'conversation'),
    ('', 'empty'),
])
def test_classification_is_deterministic(message, expected):
    assert classify(message) == expected
    assert classify(message) == expected


def _math_tool(arguments):
    return {'ok': True, 'result': '3*x**2 + 2', 'steps': ['apply power rule'],
            'verified_by': 'sympy', 'sympy_version': '1.13.1', 'summary': 'symbolic'}


def _forecast_tool(arguments):
    from helix.numerics import linear_forecast
    return linear_forecast(arguments['series'], horizon=arguments.get('horizon', 2))


def _search_tool(arguments):
    index = RetrievalIndex.ingest(FIXTURES, chunk_chars=240, overlap=40)
    return {'summary': 'ok', 'results': [chunk.to_dict() for chunk in index.search(arguments['query'], k=2)]}


def _explode(arguments):
    raise RuntimeError('tool blew up')


def test_math_plan_runs_the_tool_and_attributes_the_result():
    orchestrator = Orchestrator({'symbolic_math': _math_tool})
    result = orchestrator.run('Differentiate x**3 + 2*x', request_id='req-1', expression='x**3+2*x')
    assert result.classification == 'math'
    assert result.complied == ['symbolic_math']
    assert '3*x**2 + 2' in result.answer
    assert 'Computed by sympy' in result.answer
    assert result.tool_calls[0].ok is True
    assert result.tool_calls[0].permission == 'read-only'
    assert len(result.tool_calls[0].input_sha256) == 64
    assert 'x**3' not in json.dumps(result.to_dict()['tool_calls']), 'raw input must not be logged'


def test_forecast_plan_reports_interval_and_baseline():
    orchestrator = Orchestrator({'linear_forecast': _forecast_tool})
    result = orchestrator.run('forecast the trend', request_id='req-2',
                              series=[float(2 * i + 1) for i in range(20)], horizon=2)
    assert result.classification == 'forecast'
    assert 'prediction interval' in result.answer
    assert 'baseline' in result.answer
    assert 'helix.numerics' in result.answer
    assert result.limitations


def test_forecast_without_a_series_does_not_invent_numbers():
    calls = []
    orchestrator = Orchestrator({'linear_forecast': lambda args: calls.append(args) or {}})
    result = orchestrator.run('predict next year', request_id='req-3')
    assert 'needs at least three numeric points' in result.answer
    assert calls == []
    assert result.tool_calls == []


def test_search_without_an_index_states_the_limitation():
    orchestrator = Orchestrator({'document_search': _search_tool}, documents_available=False)
    result = orchestrator.run('find sources about attention', request_id='req-4')
    assert 'No local document index' in result.answer
    assert 'Web search is not available' in result.answer
    assert result.evidence == []
    assert result.tool_calls == []


def test_search_with_an_index_cites_every_passage():
    orchestrator = Orchestrator({'document_search': _search_tool}, documents_available=True)
    result = orchestrator.run('find sources about attention masking', request_id='req-5')
    assert result.evidence
    assert result.evidence[0]['source'] == 'transformer-notes.md'
    assert '[1] transformer-notes.md#' in result.answer
    assert all('citation' in item for item in result.evidence)


def test_code_execution_requires_explicit_approval_and_never_runs_otherwise():
    calls = []
    orchestrator = Orchestrator({'code_execution': lambda args: calls.append(args) or {'ok': True}},
                                sandbox_available=True)
    refused = orchestrator.run('run this python script', request_id='req-6', approved=False)
    assert calls == []
    assert refused.refused and 'explicit approval' in refused.refused[0]
    assert 'Nothing was executed' in refused.answer

    allowed = orchestrator.run('run this python script', request_id='req-7', approved=True)
    assert len(calls) == 1
    assert 'code_execution' in allowed.complied


def test_code_execution_without_isolation_refuses_even_when_approved():
    calls = []
    orchestrator = Orchestrator({'code_execution': lambda args: calls.append(args) or {'ok': True}},
                                sandbox_available=False)
    result = orchestrator.run('run this python script', request_id='req-8', approved=True)
    assert calls == []
    assert any('no isolation backend' in item for item in result.refused)


def test_tool_failure_is_reported_without_fabricating_a_result():
    orchestrator = Orchestrator({'symbolic_math': _explode})
    result = orchestrator.run('Differentiate x', request_id='req-9', expression='x')
    assert result.tool_calls and result.tool_calls[0].ok is False
    assert any('failed' in item for item in result.limitations)
    assert 'could not evaluate' in result.answer or 'unavailable' in result.answer


def test_conversation_without_a_model_is_honest():
    orchestrator = Orchestrator({}, generate=None)
    result = orchestrator.run('Write a poem about the sea', request_id='req-10')
    assert result.classification == 'conversation'
    assert result.needs_model is True
    assert 'no language model loaded' in result.answer
    assert 'language model unavailable' in result.limitations


def test_conversation_uses_the_model_when_available():
    orchestrator = Orchestrator({}, generate=lambda prompt: 'a generated answer')
    result = orchestrator.run('Write a poem about the sea', request_id='req-11')
    assert result.answer == 'a generated answer'
    assert 'language_model' in result.complied


def test_registered_tools_must_declare_a_permission():
    with pytest.raises(ValueError):
        Orchestrator({'arbitrary_shell': lambda args: {}})


def test_empty_message_is_rejected_explicitly():
    result = Orchestrator({}).run('   ', request_id='req-12')
    assert result.classification == 'empty'
    assert 'Send a question' in result.answer


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'resolve_data_root', lambda *args, **kwargs: tmp_path)
    app = web.create_app(config_path=str(tmp_path / 'missing.json'),
                         llm_checkpoint=str(tmp_path / 'missing.pt'))
    app.testing = True
    return app.test_client()


def test_agent_endpoint_runs_a_real_symbolic_calculation(client):
    response = client.post('/api/agent', json={
        'message': 'Differentiate x**2', 'expression': 'x**2', 'operation': 'differentiate',
    })
    assert response.status_code == 200
    body = response.get_json()
    assert body['classification'] == 'math'
    assert body['tool_calls'][0]['tool'] == 'symbolic_math'
    assert '2*x' in body['answer']
    assert body['plan']
    assert 'notice' in body


def test_agent_endpoint_reports_missing_index_and_model_honestly(client):
    search = client.post('/api/agent', json={'message': 'find sources about attention'}).get_json()
    assert 'No local document index' in search['answer']
    chat = client.post('/api/agent', json={'message': 'tell me a story'}).get_json()
    assert chat['needs_model'] is True
    assert 'no language model loaded' in chat['answer']


def test_agent_endpoint_refuses_unapproved_code(client):
    body = client.post('/api/agent', json={'message': 'run this python script'}).get_json()
    assert body['tool_calls'] == []
    assert body['refused']
    assert 'Nothing was executed' in body['answer']


def test_agent_endpoint_validates_input(client):
    assert client.post('/api/agent', json={'message': ''}).status_code == 400
    assert client.post('/api/agent', json={'message': 'x', 'series': 'nope'}).status_code == 400
    assert client.post('/api/agent', json={'message': 'x' * 5000}).status_code == 400


def test_forecast_endpoint_returns_verified_tool_output(client):
    response = client.post('/api/research/forecast', json={
        'series': [float(i * 3 + 1) for i in range(12)], 'horizon': 3, 'interval': 0.9,
    })
    assert response.status_code == 200
    body = response.get_json()
    assert body['backend'] in ('fortran', 'python')
    assert len(body['predictions']) == 3
    assert body['holdout']['beats_baseline'] is True
    assert 'not a verified prediction' in body['notice']


def test_forecast_endpoint_accepts_csv_and_rejects_bad_series(client):
    csv = 'step,value\n' + '\n'.join(f'{i},{i * 2 + 1}' for i in range(10))
    response = client.post('/api/research/forecast', json={'csv': csv, 'column': 'value', 'horizon': 2})
    assert response.status_code == 200
    assert response.get_json()['points_used'] == 10
    assert client.post('/api/research/forecast', json={'series': [1, 2]}).status_code == 400
    assert client.post('/api/research/forecast', json={'series': ['a', 'b', 'c']}).status_code == 400
    assert client.post('/api/research/forecast', json={'csv': csv, 'column': 'missing'}).status_code == 400
