# ContactDoc

Generate LLM training documents from [AlphaFold Database](https://alphafold.ebi.ac.uk/) protein structures. Each document encodes a protein's residue sequence and 3D contact map as a structured text format suitable for language model training.

Example output document:

```
<deterministic-positives-only>
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

Each document begins with a **task token** identifying the generation scheme. Documents in a shard are separated by `<end_of_document>`. Multiple generation schemes can be added via the plugin architecture (see `contactdoc/generators/`).

Contacts are heavy-atom pairs within a distance cutoff (default 4.0 A), one per residue pair, sorted by decreasing sequence separation. Leakage-resistant train/val/test splits are enforced using precomputed sequence-similarity clusters (Foldseek AFDB50).

The pipeline:

1. **Select** AFDB entries via BigQuery (filter by pLDDT, sequence length, fragment status)
2. **Download** mmCIF structures from GCS into a **Parquet dataset** (local cache with splits)
3. **Generate** documents from Parquet — parse with Gemmi, compute contacts, serialize (no GCS needed)
4. **Tokenize** into HuggingFace Arrow format for training

The Parquet intermediate (step 2) stores raw mmCIF text with metadata and cluster-based splits. Once downloaded, you can re-run document generation with different parameters (cutoffs, contact limits, etc.) without re-downloading from GCS.

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
wget -P data/ https://afdb-cluster.steineggerlab.workers.dev/v3/7-AFDB50-repId_memId.tsv.gz
wget -P data/ https://afdb-cluster.steineggerlab.workers.dev/v3/5-allmembers-repId-entryId-cluFlag-taxId.tsv.gz
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

### Stage 2: Download CIFs to Parquet

Download all mmCIF files from GCS and store them in sharded Parquet files with metadata and split assignments:

```bash
uv run python scripts/run_local.py \
  --stage download \
  --manifest-dir output/manifests \
  --output-dir output/parquet \
  --workers 32
```

This creates one Parquet file per manifest shard. Each row contains:

| Column | Type | Description |
|--------|------|-------------|
| `entry_id` | string | AFDB entry ID (e.g. `AF-A0A1C0V126-F1`) |
| `uniprot_accession` | string | UniProt accession |
| `tax_id` | int64 | NCBI taxonomy ID |
| `organism_name` | string | Scientific name |
| `global_plddt` | float32 | Global mean pLDDT |
| `seq_len` | int32 | Sequence length |
| `seq_cluster_id` | string | AFDB50 sequence cluster representative |
| `struct_cluster_id` | string | Structural cluster representative |
| `split` | string | `train`, `val`, or `test` |
| `gcs_uri` | string | Original GCS URI |
| `cif_content` | string | Raw mmCIF file text |

The Parquet dataset is the canonical local cache — all downstream analyses read from it.

### Stage 3: Generate Documents from Parquet

Generate ContactDoc documents from the local Parquet dataset (no GCS access needed):

```bash
uv run python scripts/run_local.py \
  --stage generate \
  --config config/default.yaml \
  --parquet-dir output/parquet \
  --output-dir output/results \
  --scheme deterministic-positives-only \
  --workers 32 \
  --skip-existing
```

The `--scheme` flag selects the document generation scheme (required). Each scheme is implemented as a generator plugin in `contactdoc/generators/`. The output is written to a subdirectory named after the scheme (e.g. `output/results/deterministic-positives-only/`).

Input parquet shards are grouped 10:1 into output shards — every 10 input shards produce 1 output shard. The parquet directory is searched recursively, so subdirectories are supported.

To process specific input shards into a single output shard (useful for debugging):

```bash
uv run python scripts/generate_docs.py \
  --config config/default.yaml \
  --parquet-shard output/parquet/shard_000000.parquet \
  --parquet-shard output/parquet/shard_000001.parquet \
  --shard-index 0 \
  --output-dir output/results/deterministic-positives-only \
  --scheme deterministic-positives-only
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

- **txt.gz** — concatenated documents, each ending with `<end>\n`, separated by `<end_of_document>\n`
- **metadata.jsonl.gz** — per-document metadata (entry ID, pLDDT, contact count, split, SHA1 of document text, etc.)
- **errors.jsonl.gz** — entries that were skipped (parse failures, no contacts after filtering, etc.)

### Legacy: One-Step Process (download + generate combined)

The original single-step pipeline is still available:

```bash
uv run python scripts/run_local.py \
  --stage process \
  --config config/default.yaml \
  --manifest-dir output/manifests \
  --output-dir output/results \
  --use-gcs --workers 16
```

### Cluster-Based Splits

The pipeline **requires** both precomputed cluster files for split assignment. Entries not found in either file are excluded from the corpus entirely — there is no fallback to singleton clusters. This guarantees clean train/val/test splits with no leakage between structurally or sequence-similar proteins.

Split assignment uses the structural cluster representative as the hash key, so all proteins sharing a fold land in the same split. The default config expects both files in `data/` (see Step 0).

### Stage 3: Tokenize into HuggingFace Arrow Dataset

Convert the sharded text output into a pre-tokenized HuggingFace datasets Arrow format for efficient random-access during training:

```bash
uv run python scripts/build_dataset.py \
  --input-dir output/results \
  --output-dir output/dataset \
  --max-contacts-ratio 1.0
```

`--max-contacts-ratio` limits the number of contacts per document to a multiple of the sequence length (e.g. `1.0` means at most `seq_len` contacts). When truncated, the `<end_contacts>` and `<end>` tokens are omitted so the model learns to generate contacts until the context window is full. Omit the flag to keep all contacts.

The output is one directory per split, each a HuggingFace `Dataset` with columns: `input_ids` (list[int]), `entry_id`, `seq_len`, `contacts_emitted`, `global_plddt`.

To inspect decoded documents:

```bash
uv run python scripts/view_dataset.py --dataset-dir output/dataset/train --random 3
```

### Retrying Failed Shards

If shard processing is interrupted (e.g. GCP auth expires), you can retry only the failed shards without redoing successful ones:

```bash
# Retry from a specific shard index onward
uv run python scripts/run_local.py \
  --config config/default.yaml \
  --manifest-dir output/manifests \
  --output-dir output/results \
  --use-gcs --workers 32 \
  --retry-from 3296

# Or retry specific shard indices listed in a file (one per line)
uv run python scripts/run_local.py \
  --config config/default.yaml \
  --manifest-dir output/manifests \
  --output-dir output/results \
  --use-gcs --workers 32 \
  --retry-list failed_shards.txt
```

### Uploading Parquet Dataset to HuggingFace

To share the Parquet dataset (the CIF cache with metadata and splits), first generate a manifest and then upload:

```bash
# Generate manifest with per-shard statistics
uv run python scripts/make_parquet_manifest.py \
  --parquet-dir output/parquet \
  --output output/parquet/manifest.jsonl

# Authenticate with HuggingFace
huggingface-cli login

# Upload parquet shards, manifest, and dataset card
uv run python scripts/upload_to_hf.py \
  --parquet-dir output/parquet \
  --manifest output/parquet/manifest.jsonl \
  --dataset-card DATASET_CARD.md \
  --repo-id YOUR_USERNAME/afdb-structures
```

The upload uses `upload_large_folder` which handles chunked uploads and automatic resumption for large datasets. The manifest (`manifest.jsonl`) contains a summary line followed by one record per shard with row count, file size, pLDDT statistics, sequence length range, and split distribution.

## Reproducing the v1 Corpus (~24M documents)

The commands below reproduce the full corpus on a machine with >=64GB RAM (for cluster maps) and GCP access.

```bash
# 0. Download cluster files into data/
#    File 7: 7-AFDB50-repId_memId.tsv.gz                    (1.2 GB)
#    File 5: 5-allmembers-repId-entryId-cluFlag-taxId.tsv.gz (1.6 GB)
#    From: https://afdb-cluster.steineggerlab.workers.dev/ (Version 3)

# 1. Authenticate with GCP
gcloud auth application-default login

# 2. Build manifest (streams ~178M BigQuery rows, keeps ~24M with cluster membership)
#    Takes ~5 hours. Requires ~41GB RAM for cluster maps.
uv run python scripts/bq_make_manifest.py \
  --config config/default.yaml \
  --output-dir /data/tim/contactdoc/manifests

# 3. Download CIFs to Parquet (local cache of all structures with splits)
#    Takes ~18-40 hours depending on network speed. GCP auth may expire mid-run.
uv run python scripts/run_local.py \
  --stage download \
  --manifest-dir /data/tim/contactdoc/manifests \
  --output-dir /data/tim/contactdoc/parquet \
  --workers 32

# 4. Generate documents from Parquet (no GCS needed, CPU-bound, much faster)
#    --scheme selects the generator plugin and names the output subdirectory
#    --skip-existing skips output shards already generated (safe to re-run)
#    Input shards are grouped 10:1 into output shards (~1,200 output shards)
uv run python scripts/run_local.py \
  --stage generate \
  --config config/default.yaml \
  --parquet-dir /data/tim/contactdoc/parquet \
  --output-dir /data/tim/contactdoc/results \
  --scheme deterministic-positives-only \
  --workers 32 \
  --skip-existing

# 5. Tokenize into Arrow dataset (contacts capped at 1x sequence length)
uv run python scripts/build_dataset.py \
  --input-dir /data/tim/contactdoc/results/deterministic-positives-only \
  --output-dir /data/tim/contactdoc/dataset \
  --max-contacts-ratio 1.0

# 6. (Optional) Upload Parquet dataset to HuggingFace
#    First generate a manifest, then upload
uv run python scripts/make_parquet_manifest.py \
  --parquet-dir /data/tim/contactdoc/parquet \
  --output /data/tim/contactdoc/parquet/manifest.jsonl

huggingface-cli login
uv run python scripts/upload_to_hf.py \
  --parquet-dir /data/tim/contactdoc/parquet \
  --manifest /data/tim/contactdoc/parquet/manifest.jsonl \
  --dataset-card DATASET_CARD.md \
  --repo-id timodonnell/afdb-structures
```

**Note:** GCP application-default credentials expire after ~1 hour of inactivity or ~12 hours total. If the download stage fails partway through due to auth expiry, re-authenticate and use `--retry-list` or `--retry-from` to resume (see above). The generate stage reads from local Parquet and does not need GCP credentials.

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
