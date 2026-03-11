"""Base class for document generators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..cif_parse import ParseResult
from ..contacts import Contact


@dataclass
class GeneratorResult:
    """Result of generating a document from a parsed structure."""
    doc_text: str
    contacts_pre_filter: int
    contacts_emitted: int


class DocumentGenerator(ABC):
    """Abstract base class for document generation schemes.

    Subclasses implement a specific strategy for turning a parsed protein
    structure into a text document.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used as task token and scheme directory name."""
        ...

    @abstractmethod
    def generate(self, parse_result: ParseResult, cfg) -> GeneratorResult | str:
        """Generate a document from a parsed structure.

        Returns a GeneratorResult on success, or an error reason string on failure.
        """
        ...
