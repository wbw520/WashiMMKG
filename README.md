# WashiMMKG and WikiWalk

Code for *Enhancing Washi Reasoning in MLLMs via Multimodal Knowledge Graphs and Agentic
Retrieval-Augmented Generation*.

The repository holds two things: the pipeline that builds **WashiMMKG**, a multimodal
knowledge graph of Japanese handmade paper, and **WikiWalk**, the agentic retrieval method
evaluated on it. The graph, the benchmark and the measured results are distributed
separately (see *Data* below) because they are large and because the source material has
its own terms of use.

## Layout

| Path | What it is |
|---|---|
| `washi_kg/` | Graph construction: Extractor → Filter → Detector → Creator |
| `wikirag/` | Retrieval and evaluation: the walk, the baselines, the judge |
| `bench/` | Benchmark construction (templates, verbalisation, validation, split) |
| `tools/` | Cleaning, figures, table filling, number checking, supplement |
| `build_kg.py` | Entry point for construction and export |
| `config.yaml` | Thresholds, model names and paths — nothing is hard-coded |

`wikirag/` is the part a reader of the paper most likely wants:

- `page.py` renders an entity as the page the navigator observes
- `walker.py` does root initialisation, the cite/follow split and chain verification
- `pipeline.py` merges the chains and calls the answerer
- `baselines.py` holds Direct, Know, Know⁺, FMT-RAG, GraphRAG and ReAct
- `evaluate.py` runs any method over a split and writes one result file

## Reproducing the paper

Serve a backbone with an OpenAI-compatible endpoint (vLLM is what we used), then:

```bash
# the reported configuration
python -m wikirag.evaluate --method wikiwalk --split test \
  --no-captions --hide-answer-captions --beam 5 --max-depth 4 --budget 16 \
  --base-url http://127.0.0.1:8100/v1 --model gemma-4-31b-it \
  --clip-device cuda:0 --out results/wikiwalk.test.json

# a baseline, same evidence budget
python -m wikirag.evaluate --method react --split test \
  --no-captions --hide-answer-captions --budget 16 \
  --base-url http://127.0.0.1:8100/v1 --model gemma-4-31b-it \
  --clip-device cuda:0 --out results/react.test.json

# one fixed judge re-grades a finished run without re-running retrieval
python tools/rejudge.py --url http://127.0.0.1:8100/v1 --model gemma-4-31b-it \
  --only 'react.test.*.json' --workers 16
```

`tools/fill_paper.py` writes measured numbers into the manuscript, and
`tools/check_paper_numbers.py` verifies that every number in the tables still matches the
result file it came from. `tools/make_supplement.py` regenerates the supplementary tables.

## Evaluation protocol

Two settings decide what a run means and are worth stating explicitly:

- `--no-captions` blanks every image caption, in the graph and in the benchmark records.
  All reported results use it. Showing captions raises accuracy by 0.0120, almost entirely
  by leaking the answer on multimodal-chain questions, where accuracy goes from 0.268 to
  0.416 while every other template is unchanged.
- `--budget K` is the number of triples handed to the answerer, and is the same for every
  retrieval method so the comparison isolates *which* evidence a method finds rather than
  how much it may show.

Grading uses one fixed judge (Gemma-4-31B) for every backbone. Letting each backbone grade
its own answers mixes self-preference with the weaker models' grading ability, and the two
cannot be separated afterwards.

## Data

The graph, the benchmark, the images and the measured result files are in the data
archive that accompanies the paper. Images that we cannot redistribute are listed in the
archive's manifest with a recovered source URL where one exists, and

```bash
python tools/place_images.py --data ../washi-data
```

puts all of them where the graph expects: it copies the shipped ones out of the archive
and downloads the rest, verifying each against the checksum recorded when the image was
first collected. Text-only reproduction does not need them; the multimodal and
multimodal-chain templates do. Point the code at them by setting the paths in
`config.yaml`; `WASHI_RESULTS` selects which results directory the reporting tools read.

## Citation

Please cite the paper. A BibTeX entry will be added here on publication.
