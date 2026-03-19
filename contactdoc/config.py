import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class FiltersConfig:
    skip_fragments: bool = True
    global_mean_plddt_min: float = 70.0
    residue_plddt_min: float = 70.0
    max_seq_len: int = 2048
    require_single_chain: bool = True
    canonical_residue_policy: str = "map_to_unk"


@dataclass
class ContactsConfig:
    cutoff_angstrom: float = 4.0
    exclude_adjacent_residues: bool = True
    heavy_atoms_only: bool = True
    max_contacts_per_doc: int = 2048
    tie_break: str = "lex_atom_names"


@dataclass
class SplitsConfig:
    mode: str = "afdb50"
    seed: str = "contactdoc-v1"
    train_frac: float = 0.98
    val_frac: float = 0.01
    test_frac: float = 0.01


@dataclass
class ClusterFilesConfig:
    afdb50_rep_mem_tsv_gz: str = ""
    structural_rep_mem_tsv_gz: str = ""


@dataclass
class ParallelismConfig:
    shard_size_entries: int = 2000
    num_workers_local: int = 16
    use_modal: bool = False


@dataclass
class Random3BinsConfig:
    distance_bins: list = field(default_factory=lambda: [4.0, 12.0])
    bin_sample_rates: list = field(default_factory=lambda: [1.0, 0.2, 0.1])
    false_contact_lambda: float = 2.0
    correction_resample_prob: float = 0.01
    atom_resample_prob: float = 0.01
    max_tokens: int = 8000
    long_range_threshold: int = 8
    plddt_bin_edges: list = field(default_factory=lambda: [70, 75, 80, 85, 90, 95])


@dataclass
class PipelineConfig:
    afdb_version: int = 4
    bigquery_table: str = "bigquery-public-data.deepmind_alphafold.metadata"
    gcs_bucket_prefix: str = "gs://public-datasets-deepmind-alphafold-v4/"
    output_prefix: str = "./output/contactdoc/v1/"
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    contacts: ContactsConfig = field(default_factory=ContactsConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    cluster_files: ClusterFilesConfig = field(default_factory=ClusterFilesConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    random_3_bins: Random3BinsConfig = field(default_factory=Random3BinsConfig)


def load_config(path: str | Path) -> PipelineConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _dict_to_config(raw)


def _dict_to_config(d: dict) -> PipelineConfig:
    return PipelineConfig(
        afdb_version=d.get("afdb_version", 4),
        bigquery_table=d.get("bigquery_table", PipelineConfig.bigquery_table),
        gcs_bucket_prefix=d.get("gcs_bucket_prefix", PipelineConfig.gcs_bucket_prefix),
        output_prefix=d.get("output_prefix", PipelineConfig.output_prefix),
        filters=FiltersConfig(**d.get("filters", {})),
        contacts=ContactsConfig(**d.get("contacts", {})),
        splits=SplitsConfig(**d.get("splits", {})),
        cluster_files=ClusterFilesConfig(**d.get("cluster_files", {})),
        parallelism=ParallelismConfig(**d.get("parallelism", {})),
        random_3_bins=Random3BinsConfig(**d.get("random_3_bins", {})),
    )


def config_sha(cfg: PipelineConfig) -> str:
    """Compute a deterministic SHA256 of the config for output path tagging."""
    canonical = json.dumps(_config_to_dict(cfg), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _config_to_dict(cfg: PipelineConfig) -> dict:
    from dataclasses import asdict
    return asdict(cfg)
