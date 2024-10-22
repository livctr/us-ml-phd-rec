# U.S. ML PhD Recomendation System

Disclaimer: results are not 100% accurate and there is likely some bias to how papers / professors are filtered.

### Data Pipeline

First, a list of authors are gathered from recent conference proceedings. A batched RAG pipeline is used to determine which persons are U.S. professors (unsure how accurate the LLM here is). This can be reproduced as follows:

#### Repeat scrape until satisfactory

```python
# Scrape top conferences for potential U.S.-based professors, ~45 mins
python -m data_pipeline.conference_scraper
```
**Selected conferences**
- NeurIPS: 2022, 2023 
- ICML: 2023, 2024
- AISTATS: 2023, 2024
- COLT: 2023, 2024
- AAAI: 2023, 2024
- EMNLP: 2023, 2024
- CVPR: 2023, 2024

```python
# Search authors and locally store search results. Uses Bing web search API.
python -m data_pipeline.us_professor_verifier --batch_search
```

NOTE 1: you may encounter caught exceptions due to HTTPError or invalid JSON outputs from the LLM. Would suggest to run the above multiple times until results are satisfactory.

NOTE 2: This pipeline does not handle name collisions, name changes, initials.

#### Create file containing U.S. professor data

```python
# Use locally stored search results as input to an LLM.
# Sends as batches, each one waiting for the previous to finish.
python -m data_pipeline.us_professor_verifier --batch_analyze
# After some time (at most 24 hrs per batch, ~5 batches), the batch results become available for retrieval.
# Took ~1 hr for me
python -m data_pipeline.us_professor_verifier --batch_retrieve
```

#### Extract embeddings for the relevant papers
```python
# Fetch arxiv data and extract embeddings
python -m data_pipeline.paper_embeddings_extractor
```

### Run streamlit

```python

streamlit run streamlit.py

```