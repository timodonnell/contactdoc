"""Deterministic-positives-only document generator.

Baseline scheme: residue sequence + closest heavy-atom contact per residue pair
within a distance cutoff, sorted by sequence separation (longest-range first).
"""

from ..contacts import compute_contacts, filter_contacts_by_plddt, sort_and_truncate
from ..serialize import serialize_document
from .base import DocumentGenerator, GeneratorResult


class DeterministicPositivesOnly(DocumentGenerator):

    @property
    def name(self) -> str:
        return "deterministic-positives-only"

    def generate(self, parse_result, cfg):
        contacts = compute_contacts(parse_result, cfg.contacts.cutoff_angstrom)
        contacts_pre_filter = len(contacts)

        contacts = filter_contacts_by_plddt(
            contacts, parse_result, cfg.filters.residue_plddt_min,
        )
        if not contacts:
            return "no_contacts_after_filter"

        contacts = sort_and_truncate(contacts, cfg.contacts.max_contacts_per_doc)

        doc_text = serialize_document(
            parse_result.residues, contacts, task_token=self.name,
        )
        return GeneratorResult(
            doc_text=doc_text,
            contacts_pre_filter=contacts_pre_filter,
            contacts_emitted=len(contacts),
        )
