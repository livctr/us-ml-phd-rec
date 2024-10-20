"""Pulls papers from arxiv."""
from collections import defaultdict
from functools import partial
from datetime import datetime
import heapq
import json
import os
from pathlib import Path
import pickle

from datasets import Dataset
import kaggle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


arxiv_fname = "arxiv-metadata-oai-snapshot.json"

def download_arxiv_data(path = Path(".")):
    """Downloads and unzips arxiv dataset from Kaggle into the `data` subdirectory of `path`."""
    dataset = "Cornell-University/arxiv"
    data_path = path/"data"

    if not any([arxiv_fname in file for file in os.listdir(data_path)]):
        kaggle.api.dataset_download_cli(dataset, path=data_path, unzip=True)
    else:
        print(f"Data already downloaded at {data_path/arxiv_fname}.")
    return data_path/arxiv_fname

def get_lbl_from_name(names):
    """Tuple (last_name, first_name, middle_name) => String 'first_name [middle_name] last_name'."""
    return [
        name[1] + ' ' + name[0] if name[2] == '' \
        else name[1] + ' ' + name[2] + ' ' + name[0]
        for name in names
    ]

def filter_arxiv_for_ml(arxiv_path, obtain_summary=False, authors_of_interest=None):
    """Sifts through downloaded arxiv file to find ML-related papers.
    
    If `obtain_summary` is True, saves a pickled DataFrame to the same directory as
    the downloaded arxiv file with the name `arxiv_fname` + `-summary.pkl`.

    If `authors_of_interest` is not None, only save ML-related papers by those authors.
    """
    ml_path = str(arxiv_path).split('.')[0]+'-ml.json'
    summary_path = str(arxiv_path).split('.')[0]+'-summary.pkl'

    ml_cats = ['cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'stat.ML']

    if obtain_summary and Path(ml_path).exists() and Path(summary_path).exists():
        print(f"File {ml_path} with ML subset of arxiv already exists. Skipping.")
        print(f"Summary file {summary_path} already exists. Skipping.")
        return
    if not obtain_summary and Path(ml_path).exists():
        print(f"File {ml_path} with ML subset of arxiv already exists. Skipping.")
        return

    if obtain_summary:
        gdf = {'categories': [], 'lv_date': []}  # global data

    if authors_of_interest:
        authors_of_interest = set(authors_of_interest)

    # Load the JSON file line by line
    with open(arxiv_path, 'r') as f1, open(ml_path, 'w') as f2:
        for line in tqdm(f1):
            # Parse each line as JSON
            try:
                entry_data = json.loads(line)
            except json.JSONDecodeError:
                # Skip lines that cannot be parsed as JSON
                continue

            # check categories and last version in entry data
            if (
                obtain_summary 
                and 'categories' in entry_data 
                and 'versions' in entry_data 
                and len(entry_data['versions']) 
                and 'created' in entry_data['versions'][-1]
            ):
                gdf['categories'].append(entry_data['categories'])
                gdf['lv_date'].append(entry_data['versions'][-1]['created'])

            # ml data
            authors_on_paper = get_lbl_from_name(entry_data['authors_parsed'])
            if ('categories' in entry_data
                and (any(cat in entry_data['categories'] for cat in ml_cats))
                and (any(author in authors_of_interest for author in authors_on_paper))
            ):
                f2.write(line)

    if obtain_summary:
        gdf = pd.DataFrame(gdf)
        gdf['lv_date'] = pd.to_datetime(gdf['lv_date'])
        gdf = gdf.sort_values('lv_date', axis=0).reset_index(drop=True)

        cats = set()
        for cat_combo in gdf['categories'].unique():
            cat_combo.split(' ')
            cats.update(cat_combo.split(' '))
        print(f'Columnizing {len(cats)} categories. ')
        for cat in tqdm(cats):
            gdf[cat] = pd.arrays.SparseArray(gdf['categories'].str.contains(cat), fill_value=0, dtype=np.int8)

        # count number of categories item is associated with
        gdf['ncats'] = gdf['categories'].str.count(' ') + 1

        # write to pickle file
        with open(summary_path, 'wb') as f:
            pickle.dump(gdf, f)

def get_professors_and_relevant_papers(us_professors, k=8, cutoff=datetime(2022, 10, 1)):
    """
    Returns a dictionary mapping U.S. professor names to a list of indices 
    corresponding to their most recent papers in `data/arxiv-metadata-oai-snapshot-ml.json`.
    This function is necessary to specify the papers we are interested in for each
    professor (e.g., the most recent papers after cutoff)

    Parameters:
    - us_professors: A list of U.S. professor names to match against.
    - k: The number of most recent papers to keep for each professor, based on 
         the first version upload date.
    - cutoff (datetime): Only considers papers published after this date 
                         (default: October 1, 2022).
    
    Returns:
    - dict: A dictionary where keys are professor names and values are lists of 
            indices corresponding to their most recent papers.
    """
    # professors to tuple of (datetime, papers)
    p2p = defaultdict(list)

    with open('data/arxiv-metadata-oai-snapshot-ml.json', 'r') as f:
        line_nbr = 1
        while True:
            line = f.readline()
            if not line: break

            try:
                ml_data = json.loads(line)
                paper_authors = get_lbl_from_name(ml_data['authors_parsed'])

                # filter the same way as in `conference_scraper.py`
                # ignore solo-authored papers and papers with more than 20 authors
                if len(paper_authors) == 1 or len(paper_authors) > 20:
                    continue

                try:
                    dt = datetime.strptime(ml_data["versions"][0]["created"], '%a, %d %b %Y %H:%M:%S %Z')
                    if dt < cutoff:
                        continue
                except (KeyError, ValueError) as e:
                    print(f"Failed to parse date: {ml_data}")
                    dt = datetime(2000, 1, 1)  # before cutoff date

                # consider if professor is first-author since we now care about semantics
                for paper_author in paper_authors:
                    if paper_author in us_professors:
                        # make a connection
                        heapq.heappush(p2p[paper_author], (dt, line_nbr))
                        if len(p2p[paper_author]) > k:
                            heapq.heappop(p2p[paper_author])
            except:
                print(f"{line}")
            line_nbr += 1
    return p2p

def gen(p2p):
    values = p2p.values()
    relevant_lines = set()
    for value in values:
        relevant_lines.update([v[1] for v in value])
    relevant_lines = sorted(list(relevant_lines))

    idx = 0
    with open('data/arxiv-metadata-oai-snapshot-ml.json', 'r') as f:
        line_nbr = 1
        while idx < len(relevant_lines):
            line = f.readline()
            if not line: break

            if line_nbr == relevant_lines[idx]:
                data = json.loads(line)
                yield {"line_nbr": line_nbr,
                       "id": data["id"],
                       "title": data["title"],
                       "abstract": data["abstract"], 
                       "authors": data["authors_parsed"]}
                idx += 1

            line_nbr += 1


class EmbeddingProcessor:
    def __init__(self, model_name: str, custom_model_name: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(custom_model_name)
        self.device = torch.device(device)
        self.model.to(self.device)
        torch.cuda.empty_cache()

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, batch):
        title_tkn, abstract_tkn = " [TITLE] ", " [ABSTRACT] "
        titles = batch["title"]
        abstracts = batch["abstract"]

        texts = [title_tkn + t + abstract_tkn + a for t, a in zip(titles, abstracts)]
        
        # Tokenize sentences
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Move embeddings to CPU and convert to list
        return embeddings.cpu().numpy().tolist()

    def process_dataset(self, dataset_path: str, save_path: str, batch_size: int = 128):
        # Load dataset
        ds = Dataset.load_from_disk(dataset_path)

        # Compute embeddings and add as a new column
        ds_with_embeddings = ds.map(lambda x: {"embeddings": self.get_embeddings(x)}, batched=True, batch_size=batch_size)

        # Save the updated dataset
        save_path = save_path
        ds_with_embeddings.save_to_disk(save_path)
        print(f"Dataset with embeddings saved to {save_path}")


def main():
    """Downloads arxiv data and extract embeddings for papers."""
    print("Downloading data...")
    arxiv_path = download_arxiv_data()
    with open('data/professor/us_professor.json', 'r') as f:
        authors_of_interest = json.load(f)
    authors_of_interest = [author['name'] for author in authors_of_interest]
    print("Filtering data for ML papers...")
    filter_arxiv_for_ml(arxiv_path, authors_of_interest=authors_of_interest)

    # professor to list of paper indices
    paper_data_path = "data/paper_embeddings/paper_data"
    print("Saving data to disk at " + paper_data_path)
    p2p = get_professors_and_relevant_papers(authors_of_interest)
    ds = Dataset.from_generator(partial(gen, p2p))
    ds.save_to_disk(paper_data_path)

    print("Extracting embeddings (use GPU if possible)...")
    # paper embeddings
    save_path = "data/paper_embeddings/all-mpnet-base-v2-embds"
    # Initialize the embedding processor with model names
    embedding_processor = EmbeddingProcessor(
        model_name='sentence-transformers/all-mpnet-base-v2',
        custom_model_name='salsabiilashifa11/sbert-paper'
    )
    # Process dataset and save with embeddings
    embedding_processor.process_dataset(paper_data_path, save_path, batch_size=128)

if __name__ == "__main__":
    main()
