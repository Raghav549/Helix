"""Bounded subprocess entrypoint for symbolic computations."""
from __future__ import annotations
import json
import resource
import sys


def main():
    # Limits are applied before importing heavy optional libraries.
    resource.setrlimit(resource.RLIMIT_CPU, (8, 9))
    memory = 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    from dataclasses import asdict
    from helix.research import SymbolicMath
    data = json.load(sys.stdin)
    operation = data.get('operation')
    expression = data.get('expression', '')
    variable = data.get('variable', 'x')
    tool = SymbolicMath()
    if operation in ('differentiate', 'integrate', 'solve'):
        result = getattr(tool, operation)(expression, variable)
    elif operation == 'simplify':
        result = tool.simplify(expression)
    elif operation == 'limit':
        result = tool.limit(expression, variable, data.get('point', '0'))
    else:
        raise ValueError('unsupported mathematical operation')
    value = asdict(result)
    value['ok'] = result.verified_by == 'sympy' and not result.result.startswith('error:')
    print(json.dumps(value))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}))
        sys.exit(1)
