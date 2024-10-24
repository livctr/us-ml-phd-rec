# U.S. ML PhD Recomendation System

## Usage

**Disclaimer**: This system should only be used for informational and exploratory purposes. **It will be out-of-date after the 2024 application cycle (applying for Fall 2025).* The data pipeline is based on data from selected ML conferences, OpenAI GPT-4o-mini, and heuristics. It is guaranteed that *IT MISSES MANY VERY WELL-QUALIFIED PROFESSORS*. Recommendations are not definitive and should not replace personal research, discussions with your professors at your current institutions, and direct communication with universities. Further, they are only a proxy for research alignment. Many other factors (e.g. work style, your career goals, location, etc.) need to be considered when making a decision about PhD programs. The system is limited in that it may have incomplete or biased data (the data pipeline is described below). Also, users should independently verify that the information provided is accurate, that their potential advisors are looking for PhD students, and that they are applying to the correct advisor at the correct institution (there are a few name collisions, which the pipeline doesn’t handle). Again, the recommendations should serve as an exploratory tool, and it is the responsibility of the user to do their due diligence and research when making decisions on where and how to apply, as well as their final decision. Good luck!

To get started, click on the streamlit demo!

To run locally, `git clone` the repo. Then, create a conda environment and run `pip install -r requirements.txt`. Run with `streamlit run USMLPhDRecommender.py`.

## Overview of Data Pipeline

### Overview:
1) Authors and papers are gathered from recent conference proceedings, listed [below](#selected-conferences). While these are intended to cover some of the major ML conferences, the list is not exhaustive, and there are a number of top conferences that are missing, which can lead to bias in the recommendations.
2) The authors are searched using the Bing web search API, and the returned search results are used as input into OpenAI GPT-4o-mini to determine whether the author is a U.S.-based professor. Sometimes, a search result may contain irrelevant information. There are also cases where one person may hold multiple titles or multiple people have the same name.
3) Recent papers of the U.S.-based professors are found via arXiv.
4) Finally, when a query is searched, the system uses semantic similarity to find similar papers and their corrsponding U.S.-based professors.

### "Mistakes" (Among things that I am aware of)
- This pipeline does not handle name collisions, name changes, or different spellings for the same person. Please click on the arXiv link and PDF to verify the institution.
- An LLM judges whether an author is a U.S.-based professor. No rigorous analysis has been done on the accuracy of this classification, but I can verify that there are mistakes.

### Current Heuristics
- Authors are filtered to those with at least 3 papers in the selected conferences (past 2 years) for which they are not first authors.
- Papers with only one author or >20 authors are ignored.
- The first author of each paper is ignored. Normally, these are students.

### Possibilities for improvement
- Better embeddings: The current methodology to embed papers simply packages the title and abstract as input to the LLM. An extra step can be done in which an LLM extracts the topic, insights, methodologies, etc. in the papers so that there is more focus on content.
- Better Model: At the time of writing, `gte-Qwen2-7B-instruct` appears to be the best at clustering in the arXiv section of the MTEB benchmark ([leaderboard](https://huggingface.co/spaces/mteb/leaderboard)). More powerful models can be used, if they can somehow also be deployed.
- Larger scope: due to limited financial resources, only a select number of conferences and professors are explored. The pipeline can be re-purposed to explore other exciting research areas outside of ML.
- Handling name collisions.

### Selected conferences
- NeurIPS: 2022, 2023 
- ICML: 2023, 2024
- AISTATS: 2023, 2024
- COLT: 2023, 2024
- AAAI: 2023, 2024
- EMNLP: 2023, 2024
- CVPR: 2023, 2024


## Reproducing the Project

The `data_pipeline` requires more packages. Please pip install them in your conda environment:

```bash
cd data_pipeline
pip install -r requirements-data-pipeline.txt
cd ..
```

```bash
# Scrape top conferences, ~45 mins (most time from AAAI)
python -m data_pipeline.conference_scraper
```

```bash
# Search authors and locally store search results. Uses Bing web search API.
python -m data_pipeline.us_professor_verifier --batch_search
```

NOTE: you may encounter caught exceptions, e.g., due to HTTPError. Would suggest to run the above multiple times until the conferences are scraped / all persons have been verified.

```bash
# Use locally stored search results as input to an LLM.
# Sends as batches, each one waiting for the previous to finish.
python -m data_pipeline.us_professor_verifier --batch_analyze
# After some time (at most 24 hrs per batch, ~5 batches), the batch results become available for retrieval.
# Took ~1 hr for me
python -m data_pipeline.us_professor_verifier --batch_retrieve
```

NOTE: I am not confident that the LLM can always produce valid JSON (it does so very frequently).

```bash
# Fetch arxiv data and extract embeddings (may need GPU)
python -m data_pipeline.paper_embeddings_extractor
```

```bash
# Run the streamlit application
streamlit run USMLPhDRecommender.py
```
