"""Document generator registry."""

from .base import DocumentGenerator, GeneratorResult
from .deterministic_positives_only import DeterministicPositivesOnly

# Registry: scheme name -> generator class
GENERATORS: dict[str, type[DocumentGenerator]] = {
    "deterministic-positives-only": DeterministicPositivesOnly,
}


def get_generator(name: str) -> DocumentGenerator:
    """Instantiate a generator by scheme name."""
    cls = GENERATORS.get(name)
    if cls is None:
        available = ", ".join(sorted(GENERATORS))
        raise ValueError(f"Unknown generator scheme: {name!r}. Available: {available}")
    return cls()
