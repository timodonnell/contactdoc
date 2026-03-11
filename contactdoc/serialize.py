"""Document text serialization + metadata/error JSONL generation."""

import hashlib
import json

from .cif_parse import ParseResult
from .contacts import Contact


def serialize_document(
    residues: list,
    contacts: list[Contact],
    task_token: str | None = None,
) -> str:
    """Serialize a single training document to text format.

    If task_token is provided, the document starts with <task_token_name>.
    """
    lines = []
    if task_token is not None:
        lines.append(f"<{task_token}>")
    lines.append("<begin_sequence>")

    seq_tokens = " ".join(f"<{r.name}>" for r in residues)
    lines.append(seq_tokens)

    lines.append("<begin_contacts>")
    for c in contacts:
        lines.append(f"<p{c.i}> <p{c.j}> <{c.atom_i}> <{c.atom_j}>")
    lines.append("<end_contacts>")
    lines.append("<end>")

    return "\n".join(lines) + "\n"


def make_metadata_record(
    entry: dict,
    parse_result: ParseResult,
    contacts_pre_filter: int,
    contacts_emitted: int,
    doc_text: str,
    config: object,
) -> dict:
    """Build metadata JSONL record for one emitted document."""
    residues_passing = sum(
        1 for r in parse_result.residues
        if r.plddt >= config.filters.residue_plddt_min
    )
    return {
        "entryId": entry["entryId"],
        "uniprotAccession": entry.get("uniprotAccession", ""),
        "taxId": entry.get("taxId", 0),
        "organismScientificName": entry.get("organismScientificName", ""),
        "latestVersion": entry.get("latestVersion", 0),
        "globalMetricValue_mean_pLDDT": entry.get("globalMetricValue", 0.0),
        "uniprotStart": entry.get("uniprotStart", 0),
        "uniprotEnd": entry.get("uniprotEnd", 0),
        "seq_len": len(parse_result.residues),
        "global_mean_plddt_min": config.filters.global_mean_plddt_min,
        "residue_plddt_min": config.filters.residue_plddt_min,
        "contact_cutoff_angstrom": config.contacts.cutoff_angstrom,
        "max_contacts_per_doc": config.contacts.max_contacts_per_doc,
        "contacts_found_pre_confidence_filter": contacts_pre_filter,
        "contacts_emitted": contacts_emitted,
        "residues_passing_plddt": residues_passing,
        "split": entry.get("split", ""),
        "seq_cluster_id": entry.get("seq_cluster_id", ""),
        "struct_cluster_id": entry.get("struct_cluster_id", ""),
        "split_cluster_id": entry.get("split_cluster_id", ""),
        "source_cif_gcs_uri": entry.get("gcs_uri", ""),
        "sha1_of_document_text": hashlib.sha1(doc_text.encode()).hexdigest(),
    }


def make_error_record(
    entry: dict,
    reason: str,
    exception: str | None = None,
) -> dict:
    """Build error JSONL record for a skipped/failed entry."""
    return {
        "entryId": entry.get("entryId", "unknown"),
        "gcs_uri": entry.get("gcs_uri", ""),
        "reason": reason,
        "exception": exception,
        "globalMetricValue": entry.get("globalMetricValue"),
        "seq_len": entry.get("seq_len"),
    }


def metadata_to_jsonl(record: dict) -> str:
    return json.dumps(record, ensure_ascii=True, sort_keys=False)


def error_to_jsonl(record: dict) -> str:
    return json.dumps(record, ensure_ascii=True, sort_keys=False)
