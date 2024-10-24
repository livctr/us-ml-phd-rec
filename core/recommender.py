from collections import Counter, defaultdict
import json

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from data_pipeline.config import DataPaths


class EmbeddingProcessor:
    def __init__(self,
                 model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                 custom_model_name: str = 'salsabiilashifa11/sbert-paper'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(custom_model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
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


class Recommender:
    def __init__(self,
                 embedding_processor: EmbeddingProcessor,
                 ita_path: str = DataPaths.FRONTEND_ITA_PATH,
                 weights_path: str = DataPaths.FRONTEND_WEIGHTS_PATH,
                 frontend_us_professor_path: str = DataPaths.FRONTEND_PROF_PATH,
    ):
        self.embedding_processor = embedding_processor
        self.ita = pd.read_csv(ita_path)
        self.embds = torch.load(weights_path, weights_only=True)
        # dictionary with professor names as keys and their metadata as values
        with open(frontend_us_professor_path, 'r') as f:
            self.us_professor_profiles = json.load(f)
    
    def get_top_k(self, query: str, top_k: int = 5):
        """Returns the top indices of papers most similar to the query."""
        query_batch = {'title': [query], 'abstract': [""]}
        query_embd = torch.Tensor(self.embedding_processor.get_embeddings(query_batch)[0])
        sim = self.embds @ query_embd
        return torch.argsort(sim, descending=True)[:top_k]

    def get_recommended_data(self, top_indices: torch.Tensor):
        """Returns a list of dictionaries with professors corresponding to their information."""
        selected = self.ita.iloc[top_indices]

        professors = [x.split("|-|") for x in selected["authors"]]
        professors = [prof for profs in professors for prof in profs]

        # rank professors first by number of times appeared in the list
        # and then by their order of appearance
        counts = Counter(professors)
        ranked_professors = sorted(counts.keys(), key=lambda name: (-counts[name], professors.index(name)))

        # professor to IDs
        professor2ids = defaultdict(list)
        for pid_, pt, pauthors in zip(
            selected['id'].tolist(),
            selected['title'].tolist(),
            selected['authors'].tolist()
        ):
            for prof in pauthors.split("|-|"):
                professor2ids[prof].append((pid_, pt))

        # Build professor metadata
        data = []
        for prof in ranked_professors:
            data.append({
                "name": prof,
                "title": self.us_professor_profiles[prof]["title"],
                "department": self.us_professor_profiles[prof]["department"],
                "university": self.us_professor_profiles[prof]["university"],
                "papers": professor2ids[prof],
            })
        return data


if __name__ == "__main__":
    embedding_processor = EmbeddingProcessor()
    recommender = Recommender(embedding_processor)

    top_k = recommender.get_top_k("What is the most important aspect of machine learning in computer science?", top_k=10)
    data = recommender.get_recommended_data(top_k)
    print(data)
