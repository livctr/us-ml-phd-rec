import os

class DataPaths:

    BASE_DIR = "data"
    LOG_DIR = "logs"

    PROGRESS_LOG_PATH = os.path.join(LOG_DIR, 'progress_log.tmp')

    CONFERENCE_DIR = os.path.join(BASE_DIR, 'conference')
    AUTHORS_PATH = os.path.join(CONFERENCE_DIR, 'authors.txt')

    PROF_DIR = os.path.join(BASE_DIR, 'professor')
    SEARCH_RESULTS_PATH = os.path.join(PROF_DIR, 'search_results.json')
    US_PROF_PATH = os.path.join(PROF_DIR, 'us_professor.json')
    NOT_US_PROF_PATH = os.path.join(PROF_DIR, 'not_us_professor.json')
    PROMPT_DATA_PREFIX = str(os.path.join(PROF_DIR, 'prompt_data'))

    ARXIV_FNAME = 'arxiv-metadata-oai-snapshot.json'
    ARXIV_PATH = os.path.join(BASE_DIR, ARXIV_FNAME)
    ML_ARXIV_PATH = os.path.join(BASE_DIR, 'arxiv-metadata-oai-snapshot-ml.json')

    PAPER_DIR = os.path.join(BASE_DIR, "paper_embeddings")
    EMBD_MODEL = "all-mpnet-base-v2-embds"
    EMBD_PATH = os.path.join(PAPER_DIR, EMBD_MODEL)
    PAPER_DATA_PATH = os.path.join(PAPER_DIR, "paper_data")

    FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend_data')
    FRONTEND_PROF_PATH = os.path.join(FRONTEND_DIR, 'us_professor.json')
    FRONTEND_EMBD_PATH = os.path.join(FRONTEND_DIR, EMBD_MODEL)  # contains id, title, author, weights
    FRONTEND_ITA_PATH = os.path.join(FRONTEND_EMBD_PATH, 'id_title_author')
    FRONTEND_WEIGHTS_PATH = os.path.join(FRONTEND_EMBD_PATH, 'weights.pt')


    # create FRONTEND_DIR PROF_DIR CONFERENCE_DIR

    @staticmethod
    def ensure_directories():
        # Create the directories if they do not exist
        os.makedirs(DataPaths.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(DataPaths.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(DataPaths.MODEL_OUTPUT_DIR, exist_ok=True)

# Call this function early in your pipeline
DataPaths.ensure_directories()
