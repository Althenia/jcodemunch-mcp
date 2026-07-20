# Odysseus rag_server vs jCodeMunch -- retrieval-layer benchmark

**What this is:** for identical code-navigation queries, the tokens each
retrieval layer injects into the model's context. Complementary framing --
jCodeMunch runs *inside* Odysseus over SSE; this shows the delta of routing
code retrieval through it instead of the built-in `rag_server`.

**Odysseus pipeline reproduced from source** (`src/rag_vector.py`): embeddings `sentence-transformers/all-MiniLM-L6-v2`; chunking `_split_into_chunks` (char/sentence, size=1000 chars, overlap=200); retrieval `search(k=5)`.

**jCodeMunch figures** are from `run_benchmark.py` (`measure_jmunch`) on the
same IndexStore content -- both sides read byte-identical source.

**Read the two axes together.** Token count alone is a trap: Odysseus's
rag_server returns fixed ~1000-char fragments, so on repos with large
symbols it can inject *fewer* tokens than jCodeMunch -- but those fragments
are frequently cut mid-definition. jCodeMunch returns *complete* symbols by
construction. So compare `tokens/query` next to `complete chunks` (of 5):
RAG cheapness that comes with split chunks is truncated context, not a win.

**Caveats:** FAISS here vs ChromaDB in Odysseus (immaterial to token count);
pure-vector ranking here vs Odysseus's 0.7/0.3 hybrid (can reorder top-k, not
its token cost -- relevance reported separately). Not a live agent-loop run.

| Repo | Files | Odysseus rag tokens/q | jCodeMunch tokens/q | Token delta | Odysseus complete/5 | Odysseus split/5 | Odysseus terms-hit/5 |
|------|------:|----------------------:|--------------------:|------------:|:-------------------:|:----------------:|:--------------------:|
| expressjs/express | 172 | 1,222 | 924 | jcm **1.3x leaner** | 0.2 | 0.4 | 3.8 |
| fastapi/fastapi | 951 | 603 | 1,834 | RAG 3.0x leaner* | 0.4 | 3.0 | 5.0 |
| gin-gonic/gin | 98 | 1,306 | 1,124 | jcm **1.2x leaner** | 1.6 | 0.6 | 4.8 |

\* *Where RAG shows fewer tokens, check its complete/5 and split/5: the saving comes from truncated ~1000-char fragments, while jCodeMunch's tokens are whole symbols. Cheaper context that is cut mid-function is not cheaper to reason over.*

## Per-query detail

### expressjs/express

*747 Odysseus chunks | 8.3s embed*

| Query | Odysseus rag tokens | Complete/5 | Split/5 | Terms-hit/5 | Query ms |
|-------|-------------------:|:----------:|:-------:|:-----------:|---------:|
| `router route handler` | 1,488 | 0/5 | 1/5 | 5/5 | 15.8 |
| `middleware` | 1,459 | 1/5 | 0/5 | 5/5 | 8.2 |
| `error exception` | 625 | 0/5 | 0/5 | 5/5 | 8.0 |
| `request response` | 1,293 | 0/5 | 0/5 | 4/5 | 8.3 |
| `context bind` | 1,243 | 0/5 | 1/5 | 0/5 | 8.1 |

### fastapi/fastapi

*5,383 Odysseus chunks | 99.03s embed*

| Query | Odysseus rag tokens | Complete/5 | Split/5 | Terms-hit/5 | Query ms |
|-------|-------------------:|:----------:|:-------:|:-----------:|---------:|
| `router route handler` | 1,219 | 1/5 | 1/5 | 5/5 | 19.4 |
| `middleware` | 630 | 1/5 | 0/5 | 5/5 | 12.2 |
| `error exception` | 317 | 0/5 | 4/5 | 5/5 | 14.5 |
| `request response` | 454 | 0/5 | 5/5 | 5/5 | 13.9 |
| `context bind` | 395 | 0/5 | 5/5 | 5/5 | 13.9 |

### gin-gonic/gin

*890 Odysseus chunks | 19.99s embed*

| Query | Odysseus rag tokens | Complete/5 | Split/5 | Terms-hit/5 | Query ms |
|-------|-------------------:|:----------:|:-------:|:-----------:|---------:|
| `router route handler` | 1,063 | 1/5 | 1/5 | 5/5 | 22.0 |
| `middleware` | 1,307 | 2/5 | 0/5 | 5/5 | 13.8 |
| `error exception` | 1,187 | 3/5 | 1/5 | 4/5 | 13.6 |
| `request response` | 1,380 | 1/5 | 0/5 | 5/5 | 13.3 |
| `context bind` | 1,591 | 1/5 | 1/5 | 5/5 | 19.4 |
