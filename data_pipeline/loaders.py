import json
import os

def load_conference_papers(conference_dir='data/conference'):
    papers = []
    files = os.listdir(conference_dir)
    for file in files:
        if not file.endswith('.json'):
            continue
        with open(os.path.join(conference_dir, file), 'r') as f:
            while True:
                line = f.readline()
                if not line: break
                paper = json.loads(line)
                papers.append(paper)
    return papers

def load_us_professor():
    """Returns a JSON list"""
    with open('data/professor/us_professor.json', 'r') as f:
        us_professors = json.load(f)
    return us_professors
