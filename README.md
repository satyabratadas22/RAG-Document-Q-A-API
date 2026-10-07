# RAG-Document-Q-A-API OR RAG DocQA

A small retrieval-augmented generation (RAG) service that answers questions from your own documents (`.pdf`, `.txt`, `.md`) and **refuses to answer when the documents don't support it**. Built with sentence-transformers, FAISS, a Hugging Face LLM and FastAPI, with a built-in evaluation script.

## How it works

```
documents -> overlapping chunks -> sentence-transformer embeddings -> FAISS (cosine search)
          -> context-window budgeting -> versioned prompt -> Hugging Face LLM (streaming)
          -> grounding check -> answer  (or abstain)
```

| Stage | Details |
|---|---|
| Chunking | 180-word windows with 40-word overlap so boundary sentences are not lost |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, L2-normalised |
| Retrieval | FAISS `IndexFlatIP` (inner product = cosine similarity), top-4 |
| Context budget | Best chunks are added until a 700-word limit is reached |
| Generation | `Qwen/Qwen2.5-0.5B-Instruct`, greedy decoding, streamed to measure TTFT and tokens/s |
| Prompts | Versioned templates (`v1`, `v2`) in `ragqa/prompts.py` |
| API | FastAPI `/ask` and `/health` endpoints |

### Reliability features
- **Retrieval threshold:** if the best chunk's similarity is below `0.30`, the system abstains without calling the LLM.
- **Grounding filter:** if fewer than 50% of the answer's content words appear in the retrieved context, the answer is replaced with an abstention. This is a cheap lexical proxy for hallucination, **not** a semantic fact-check.
- **Prompt versioning:** `v2` instructs the model to answer only from the context, cite the source and say "I don't know" otherwise.

## Project structure

```
ragqa/
  chunking.py    # chunking + document loading (pdf/txt/md)
  index.py       # embedder + FAISS index
  generate.py    # streaming HF generation with TTFT / tokens-per-second
  pipeline.py    # retrieval, context budgeting, abstention, grounding check
  prompts.py     # versioned prompts
  api.py         # FastAPI app
eval.py          # evaluation and prompt comparison
tests/           # unit tests (no model download needed)
data/docs/       # sample corpus (5 documents)
data/eval_set.json  # 20 labelled questions (15 answerable, 5 out-of-scope)
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                 # unit tests
python eval.py --prompts v1 v2      # evaluation (downloads models on first run)
uvicorn ragqa.api:app --port 8000   # start the API
```

Ask a question:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the nprobe parameter control?"}'
```

Example response fields: `answer`, `abstained`, `reason`, `sources` (with similarity scores), `grounded` and `metrics` (TTFT, tokens/s).

Use your own documents by placing files in `data/docs/` (or set `DOCS_DIR`). Choose the prompt with `PROMPT_VERSION=v1|v2`.

## Evaluation

Run on the included sample corpus (5 documents, 5 chunks) with 20 labelled questions (15 answerable, 5 out-of-scope), on a Google Colab runtime.

| Prompt | Retrieval hit@4 | ROUGE-L | Grounding score | Correct abstentions | TTFT (s) | Tokens/s |
|---|---|---|---|---|---|---|
| v1 | 0.933 | 0.318 | 0.821 | 5/5 | 0.313 | 33.2 |
| v2 | 0.933 | 0.311 | 0.872 | 5/5 | 0.313 | 32.3 |

- **ROUGE-L** compares answers with short reference answers.
- **Grounding score** is the lexical overlap described above (higher is better).
- **Correct abstentions** is the share of out-of-scope questions the system refused to answer.

## Limitations

- The sample corpus is tiny: with only 5 chunks, hit@4 is not a meaningful retrieval metric. Add more documents to evaluate retrieval properly.
- The grounding check is word overlap, so it can miss paraphrased errors and wrongly block correct paraphrases.
- ROUGE-L undervalues correct answers phrased differently from the short references.
- A 0.5B-parameter model is fast but limited; larger models would answer better.

## Possible next steps

- Larger corpus and a hit@1 / MRR retrieval metric
- Re-ranking and hybrid (BM25 + dense) retrieval
- LoRA fine-tuning of the generator
- A semantic or LLM-based faithfulness check instead of word overlap
