# ContactDoc

Generate LLM training documents from [AlphaFold Database](https://alphafold.ebi.ac.uk/) protein structures. Each document encodes a protein's residue sequence and 3D contact map as a structured text format suitable for language model training.

Example output document:

```
<begin_sequence>
<MET> <LYS> <PHE> <CYS> <ASP> <TYR> <GLY> <LEU>
<begin_contacts>
<p1> <p8> <SD> <CD1>
<p1> <p7> <CG> <CA>
<p2> <p8> <NZ> <O>
<p1> <p6> <CE> <OH>
<end_contacts>
<end>
```

Contacts are heavy-atom pairs within a distance cutoff (default 4.0 A), one per residue pair, sorted by decreasing sequence separation. Leakage-resistant train/val/test splits are enforced using precomputed sequence-similarity clusters (Foldseek AFDB50).

The pipeline:

1. **Select** AFDB entries via BigQuery (filter by pLDDT, sequence length, fragment status)
2. **Download** mmCIF structures from Google Cloud Storage
3. **Parse** with Gemmi, compute contacts via NeighborSearch/ContactSearch
4. **Serialize** to sharded, gzipped text + JSONL metadata + JSONL error logs

See [SPEC.md](SPEC.md) for the full design specification.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- A Google Cloud account with a billing-enabled project (public data access is free-tier)

## Installation

```bash
git clone <this-repo>
cd contactdoc
uv sync --all-extras
```

## GCP Setup

The pipeline reads from two public Google Cloud datasets. You need authenticated credentials tied to a GCP project (for billing attribution — actual cost is negligible).

### 1. Install the Google Cloud CLI

```bash
# Ubuntu/Debian
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
  | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install -y google-cloud-cli
```

### 2. Configure your project

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable bigquery.googleapis.com storage.googleapis.com
```

If you don't have a project yet:

```bash
gcloud projects create contactdoc --name="ContactDoc"
gcloud config set project contactdoc
```

You may need to link a billing account at https://console.cloud.google.com/billing (required even for free-tier access to public datasets).

### 3. Authenticate

```bash
gcloud auth application-default login
```

Verify everything works:

```bash
# BigQuery — should return a row count
bq query --use_legacy_sql=false \
  'SELECT COUNT(*) FROM `bigquery-public-data.deepmind_alphafold.metadata` WHERE latestVersion = 4'

# GCS — should list files
gcloud storage ls gs://public-datasets-deepmind-alphafold-v4/AF-P01308-F1-*
```

## Generating a Large Document Corpus

The pipeline runs in three steps: **download cluster data**, **build manifest** (BigQuery selection), and **process shards** (CIF download + parsing + serialization).

### Step 0: Download the Cluster Files

Both cluster files are **required** — only entries present in both are included in the corpus. This ensures every entry has proper cluster assignments for leakage-resistant train/val/test splits (no singleton fallbacks).

Download both files from the [Steinegger lab AFDB cluster page](https://afdb-cluster.steineggerlab.workers.dev/) (Version 3, which covers AFDB v4 entries):

```bash
mkdir -p data
# Download these two files into data/:
#   File 7: 7-AFDB50-repId_memId.tsv.gz                    (1.2 GB) — sequence clusters (50% identity)
#   File 5: 5-allmembers-repId-entryId-cluFlag-taxId.tsv.gz (1.6 GB) — all members with cluster flags
```

The pipeline uses **both** cluster types:
- **Sequence clusters (AFDB50, file 7)** — groups proteins at 50% sequence identity
- **Structural clusters (file 5, cluFlag=2)** — groups proteins by 3D fold similarity, which is stricter (two proteins with low sequence identity can share a fold). Only the ~30M entries with `cluFlag=2` (structurally clustered) are loaded; fragments, singletons, and sequence-only entries are excluded.

Split assignment is based on the **structural** clusters, so proteins with similar folds always land in the same split. An entry must appear in both files to be included — this yields ~30M eligible entries.

### Stage 1: Build the Manifest

Query BigQuery to select AFDB entries and write sharded manifest files:

```bash
uv run python scripts/bq_make_manifest.py \
  --config config/default.yaml \
  --output-dir output/manifests
```

With the default config this selects all AFDB v4 entries that are:
- Not fragments (full UniProt coverage)
- Global mean pLDDT >= 70
- Sequence length <= 2048
- Single polymer chain
- Present in both cluster files (AFDB50 + structural)

Entries from BigQuery that are missing from either cluster file are dropped (count logged in the output). This produces JSONL manifest shards in `output/manifests/`, each containing up to 2000 entries with GCS URI, metadata, cluster IDs, and train/val/test split assignment.

To test with a smaller set first:

```bash
uv run python scripts/bq_make_manifest.py \
  --config config/default.yaml \
  --output-dir output/manifests \
  --limit 500
```

### Stage 2: Process Manifest Shards

Process all manifest shards in parallel — this downloads each CIF from GCS, parses it with Gemmi, computes contacts, and writes gzipped output shards:

```bash
uv run python scripts/run_local.py \
  --config config/default.yaml \
  --manifest-dir output/manifests \
  --output-dir output/results \
  --use-gcs \
  --workers 16
```

Adjust `--workers` based on your machine's cores and network bandwidth. Each worker processes one manifest shard independently.

To process a single shard (useful for debugging):

```bash
uv run python scripts/process_manifest_shard.py \
  --config config/default.yaml \
  --manifest-shard output/manifests/manifest_shard_000000.jsonl \
  --shard-index 0 \
  --output-dir output/results \
  --use-gcs
```

### Output Structure

```
output/results/
  split=train/
    shard=000000.txt.gz                 # concatenated training documents
    shard=000000.metadata.jsonl.gz      # one JSON record per document
    shard=000000.errors.jsonl.gz        # skipped/failed entries
    shard=000001.txt.gz
    ...
  split=val/
    shard=000000.txt.gz
    ...
  split=test/
    shard=000000.txt.gz
    ...
```

- **txt.gz** — concatenated documents, each ending with `<end>\n`
- **metadata.jsonl.gz** — per-document metadata (entry ID, pLDDT, contact count, split, SHA1 of document text, etc.)
- **errors.jsonl.gz** — entries that were skipped (parse failures, no contacts after filtering, etc.)

### Cluster-Based Splits

The pipeline **requires** both precomputed cluster files for split assignment. Entries not found in either file are excluded from the corpus entirely — there is no fallback to singleton clusters. This guarantees clean train/val/test splits with no leakage between structurally or sequence-similar proteins.

Split assignment uses the structural cluster representative as the hash key, so all proteins sharing a fold land in the same split. The default config expects both files in `data/` (see Step 0).

## Configuration

All pipeline behavior is controlled by a YAML config file. See `config/default.yaml` for the full set of options:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `filters` | `skip_fragments` | `true` | Skip entries that don't cover the full UniProt sequence |
| `filters` | `global_mean_plddt_min` | `70.0` | Minimum global mean pLDDT to include an entry |
| `filters` | `residue_plddt_min` | `70.0` | Minimum per-residue pLDDT for a contact to be emitted |
| `filters` | `max_seq_len` | `2048` | Maximum sequence length |
| `filters` | `canonical_residue_policy` | `map_to_unk` | `map_to_unk` or `skip_entry` for non-standard residues |
| `contacts` | `cutoff_angstrom` | `4.0` | Heavy-atom distance cutoff for contacts |
| `contacts` | `max_contacts_per_doc` | `2048` | Maximum contact lines per document |
| `splits` | `train_frac` | `0.98` | Fraction of clusters assigned to train |
| `splits` | `val_frac` | `0.01` | Fraction assigned to val |
| `parallelism` | `shard_size_entries` | `2000` | Entries per manifest shard |

## Running Tests

Tests use synthetic CIF fixtures and require no GCP access:

```bash
uv run pytest tests/ -v
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `DefaultCredentialsError` | Run `gcloud auth application-default login` |
| `403 Access Denied` on BigQuery | `gcloud services enable bigquery.googleapis.com` |
| `403 Access Denied` on GCS | `gcloud services enable storage.googleapis.com` |
| `Project not found` | `gcloud config set project YOUR_PROJECT_ID` |
| Billing not enabled | Link a billing account at https://console.cloud.google.com/billing |
