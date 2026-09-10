from gpu_corruptnet.codeanalysis import analyze_source, cyclomatic_complexity, summarize

SRC = '''
def simple():
    """documented."""
    return 1


def branchy(x):
    if x > 0 and x < 10:
        for i in range(x):
            if i % 2:
                return i
    return 0


class Thing:
    def method(self):
        return 2
'''


def test_complexity_counts_branches():
    report = analyze_source(SRC, "t.py")
    by_name = {f.name: f for f in report.functions}
    assert by_name["simple"].complexity == 1
    assert by_name["branchy"].complexity >= 4  # if + and + for + if
    assert by_name["simple"].documented is True
    assert by_name["branchy"].documented is False


def test_file_level_metrics():
    report = analyze_source(SRC, "t.py")
    assert report.num_functions == 3  # simple, branchy, method
    assert report.num_classes == 1
    assert 0.0 <= report.docstring_coverage <= 1.0
    assert report.max_complexity == max(f.complexity for f in report.functions)


def test_summarize_flags_riskiest():
    s = summarize([analyze_source(SRC, "t.py")], risk_threshold=3)
    assert s["functions"] == 3
    assert s["riskiest"][0]["name"] == "branchy"
    assert any(f["name"] == "branchy" for f in s["high_risk_functions"])


def test_cyclomatic_of_flat_function_is_one():
    import ast

    node = ast.parse("def f():\n    return 1").body[0]
    assert cyclomatic_complexity(node) == 1
