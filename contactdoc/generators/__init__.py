"""Document generator registry."""

from .base import DocumentGenerator, GeneratorResult
from .contacts_and_distances_v1 import ContactsAndDistancesV1
from .deterministic_positives_only import DeterministicPositivesOnly
from .random_3_bins import Random3Bins

# Registry: scheme name -> generator class
GENERATORS: dict[str, type[DocumentGenerator]] = {
    "contacts-and-distances-v1": ContactsAndDistancesV1,
    "deterministic-positives-only": DeterministicPositivesOnly,
    "random-3-bins": Random3Bins,
}


def get_generator(name: str) -> DocumentGenerator:
    """Instantiate a generator by scheme name."""
    cls = GENERATORS.get(name)
    if cls is None:
        available = ", ".join(sorted(GENERATORS))
        raise ValueError(f"Unknown generator scheme: {name!r}. Available: {available}")
    return cls()
