"""Restricted, single-pass substitution; identical to the server contract."""

import re
from string import Formatter

VARIABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MUSTACHE = re.compile(r"{{(.*?)}}", re.DOTALL)


def compile_template(template, values, template_format):
    if template_format == "mustache":
        names = [match.group(1).strip() for match in MUSTACHE.finditer(template)]
        remainder = MUSTACHE.sub("", template)
        if "{{" in remainder or "}}" in remainder:
            raise ValueError("Unclosed or unmatched {{variable}} placeholder")
    elif template_format == "fstring":
        names = []
        for _, field, spec, conversion in Formatter().parse(template):
            if field is not None:
                if spec or conversion:
                    raise ValueError(
                        "Format specifications and conversions are not supported"
                    )
                names.append(field)
    else:
        raise ValueError("Unknown template format")
    if any(not VARIABLE.fullmatch(name) for name in names):
        raise ValueError("Variables must be simple names")
    if len(set(names)) > 100:
        raise ValueError("A prompt may contain at most 100 variables")
    missing = sorted(set(names) - values.keys())
    if missing:
        raise ValueError("Missing variables: " + ", ".join(missing))
    if any(not isinstance(value, str) for value in values.values()):
        raise ValueError("Variable values must be strings")
    compiled = (
        MUSTACHE.sub(lambda match: values[match.group(1).strip()], template)
        if template_format == "mustache"
        else template.format_map(values)
    )
    if len(compiled) > 100_000:
        raise ValueError("Compiled prompt exceeds 100,000 characters")
    return compiled
