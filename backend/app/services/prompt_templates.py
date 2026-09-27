"""Strict variable substitution. No code, attributes, loops or expressions."""
import re
from string import Formatter

VARIABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MUSTACHE = re.compile(r"{{(.*?)}}", re.DOTALL)
MAX_COMPILED_LENGTH = 100_000


def template_variables(template: str, template_format: str = "mustache") -> list[str]:
    if template_format == "mustache":
        names = [match.group(1).strip() for match in MUSTACHE.finditer(template)]
        remainder = MUSTACHE.sub("", template)
        if "{{" in remainder or "}}" in remainder:
            raise ValueError("Unclosed or unmatched {{variable}} placeholder")
    elif template_format == "fstring":
        names = []
        try:
            for _, field, spec, conversion in Formatter().parse(template):
                if field is not None:
                    if spec or conversion:
                        raise ValueError("Format specifications and conversions are not supported")
                    names.append(field)
        except ValueError as exc:
            raise ValueError(f"Invalid {{variable}} template: {exc}") from exc
    else:
        raise ValueError("Unknown template format")
    if any(not VARIABLE.fullmatch(name) for name in names):
        raise ValueError("Variables must be simple names, such as query or customer_name")
    if len(set(names)) > 100:
        raise ValueError("A prompt may contain at most 100 variables")
    return sorted(set(names))


def compile_template(template: str, values: dict[str, str], template_format: str = "mustache") -> str:
    names = template_variables(template, template_format)
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError("Missing variables: " + ", ".join(missing))
    if any(not isinstance(value, str) for value in values.values()):
        raise ValueError("Variable values must be strings")
    if template_format == "mustache":
        # Substitute once: values containing {{other}} remain literal text.
        compiled = MUSTACHE.sub(lambda match: values[match.group(1).strip()], template)
    else:
        compiled = template.format_map(values)
    if len(compiled) > MAX_COMPILED_LENGTH:
        raise ValueError("Compiled prompt exceeds 100,000 characters")
    return compiled
