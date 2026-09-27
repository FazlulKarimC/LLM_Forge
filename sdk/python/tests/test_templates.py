import pytest
from llmforge import Prompt


@pytest.mark.parametrize(
    "text,format,values,expected",
    [
        ('{"answer":"{{query}}"}', "mustache", {"query": "yes"}, '{"answer":"yes"}'),
        ("{{{query}}}", "fstring", {"query": "yes"}, "{yes}"),
        ("{query}", "fstring", {"query": "{other}"}, "{other}"),
    ],
)
def test_compilation(text, format, values, expected):
    snapshot = Prompt("v", "p", "Test", 1, text, format, ("query",))
    assert snapshot.compile(values) == expected


@pytest.mark.parametrize(
    "text,format,values",
    [
        ("{{query}}", "mustache", {}),
        ("{query.__class__}", "fstring", {"query": "x"}),
        ("{query!r}", "fstring", {"query": "x"}),
        ("{{query", "mustache", {"query": "x"}),
        ("{{query}}", "mustache", {"query": 1}),
        ("{query:>5}", "fstring", {"query": "x"}),
    ],
)
def test_invalid_compilation(text, format, values):
    with pytest.raises(ValueError):
        Prompt("v", "p", "Test", 1, text, format, ()).compile(values)
