"""Pulls papers from arxiv and gets their embeddings."""
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
from tqdm import tqdm

from core.recommender import EmbeddingProcessor
from data_pipeline.config import DataPaths

arxiv_fname = "arxiv-metadata-oai-snapshot.json"

def download_arxiv_data():
    """Downloads and unzips arxiv dataset from Kaggle into `data` directory."""
    dataset = "Cornell-University/arxiv"
    data_path = DataPaths.BASE_DIR

    if not any([DataPaths.ARXIV_FNAME in file for file in os.listdir(data_path)]):
        kaggle.api.dataset_download_cli(dataset, path=data_path, unzip=True)
    else:
        print(f"Data already downloaded at {DataPaths.ARXIV_FNAME}.")
    return DataPaths.ARXIV_FNAME

def get_lbl_from_name(names):
    """Tuple (last_name, first_name, middle_name) => String 'first_name [middle_name] last_name'."""
    return [
        name[1] + ' ' + name[0] if name[2] == '' \
        else name[1] + ' ' + name[2] + ' ' + name[0]
        for name in names
    ]

def filter_arxiv_for_ml(obtain_summary=False, authors_of_interest=None):
    """Sifts through downloaded arxiv file to find ML-related papers.
    
    If `obtain_summary` is True, saves a pickled DataFrame to the same directory as
    the downloaded arxiv file with the name `arxiv_fname` + `-summary.pkl`.

    If `authors_of_interest` is not None, only save ML-related papers by those authors.
    """
    ml_path = str(DataPaths.ARXIV_PATH).split('.')[0]+'-ml.json'
    summary_path = str(DataPaths.ARXIV_PATH).split('.')[0]+'-summary.pkl'

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
    with open(DataPaths.ARXIV_PATH, 'r') as f1, open(ml_path, 'w') as f2:
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
    corresponding to their most recent papers in DataPaths.ML_ARXIV_PATH.
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
    # professors to tuple of (datetime, arxiv_id)
    p2p = defaultdict(list)

    with open(DataPaths.ML_ARXIV_PATH, 'r') as f:
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
                    print(f"Failed to parse date: \n{ml_data}\nError: {e}")
                    dt = datetime(2000, 1, 1)  # before cutoff date

                # consider if professor is first-author since we now care about semantics
                for paper_author in paper_authors:
                    if paper_author in us_professors:
                        # make a connection
                        heapq.heappush(p2p[paper_author], (dt, ml_data["id"]))
                        if len(p2p[paper_author]) > k:
                            heapq.heappop(p2p[paper_author])
            except:
                print(f"{line}")
    return p2p

def gen(p2p):
    values = p2p.values()
    relevant_ids = set()
    for value in values:
        relevant_ids.update([v[1] for v in value])
    with open(DataPaths.ML_ARXIV_PATH, 'r') as f:
        while True:
            line = f.readline()
            if not line: break

            data = json.loads(line)
            if data["id"] in relevant_ids:
                authors = get_lbl_from_name(data["authors_parsed"])
                authors = [a for a in authors if a in p2p]  # keep authors who are U.S. professors

                yield {"id": data["id"],
                       "title": data["title"],
                       "abstract": data["abstract"], 
                       "authors": authors
                    }

def save_paper_to_professor(p2p, save_path):
    """Returns a dictionary mapping an Arxiv ID to U.S. professor names
    
    `p2p`: mapping from professor to list of paper indices in DataPaths.ML_ARXIV_PATH
    `ds`: dataset with Arxiv links and line_nbr
    """

    id2p = defaultdict(list)
    for professor, dt_and_ids in p2p.items():
        for _, id_ in dt_and_ids:
            id2p[id_].append(professor)

    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(id2p, f)
    return id2p



def main():
    """Downloads arxiv data and extract embeddings for papers."""

    ### Download and filter for ML papers written by U.S. professors ###
    # print("Downloading data...")
    # download_arxiv_data()
    # with open(DataPaths.US_PROF_PATH, 'r') as f:
    #     authors_of_interest = json.load(f)
    # authors_of_interest = [author['name'] for author in authors_of_interest]
    # print("Filtering data for ML papers...")
    # filter_arxiv_for_ml(authors_of_interest=authors_of_interest)

    # ### Create a dataset containing paper info, e.g., title, abstract, authors, etc. ###
    # print("Saving paper data to disk at " + DataPaths.PAPER_DATA_PATH)
    # p2p = get_professors_and_relevant_papers(authors_of_interest)
    # ds = Dataset.from_generator(partial(gen, p2p))
    # ds.save_to_disk(DataPaths.PAPER_DATA_PATH)

    # ### Extract paper embeddings ###
    # print("Extracting embeddings (use GPU if possible)...")
    # # Initialize the embedding processor with model names
    # embedding_processor = EmbeddingProcessor(
    #     model_name='sentence-transformers/all-mpnet-base-v2',
    #     custom_model_name='salsabiilashifa11/sbert-paper'
    # )
    # # Process dataset and save with embeddings
    # embedding_processor.process_dataset(DataPaths.PAPER_DATA_PATH, DataPaths.EMBD_PATH, batch_size=128)

    ### Create front-end data ###

    # Filter ds for paper title, id, authors, and embedding
    embds = Dataset.load_from_disk(DataPaths.EMBD_PATH)

    def join_authors(x):
        x['authors'] = "|-|".join(x['authors'])
        return x

    import pdb ; pdb.set_trace()

    embds = embds.map(join_authors)

    # save id and title to disk
    embds.select_columns(['id', 'title', 'authors']).to_csv(DataPaths.FRONTEND_ITA_PATH)
    # save embeddings as torch tensor
    embds_weights = torch.Tensor(embds['embeddings'])
    torch.save(embds_weights, DataPaths.FRONTEND_WEIGHTS_PATH)

if __name__ == "__main__":
    main()
