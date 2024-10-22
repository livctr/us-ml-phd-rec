"""Scrape data from some famous ML conferences and saves into `DataPaths.CONFERENCE_DIR`.

Every scrape function returns a list of 3-lists of the form
    [paper_title, paper_authors, paper_url].

Conferences
-----------
NeurIPS: 2022, 2023 
ICML: 2023, 2024
AISTATS: 2023, 2024
COLT: 2023, 2024
AAAI: 2023, 2024
EMNLP: 2023, 2024
CVPR: 2023, 2024
-----------

Disclaimer
-----------
The choice of conferences was sourced from here:
https://www.kaggle.com/discussions/getting-started/115799

The priority of including certain conferences and tracks was based on a 1st-year PhD's
judgment. Some very top conferences were excluded due to higher activation energy to
scrape data and/or the ignorance of the 1st-year PhD. Some notable exceptions include
ICLR, ICCV, ECCV, ACL, NAACL, and many others.
-----------
"""

from functools import partial
import json
import os
import requests
import time

from bs4 import BeautifulSoup
from tqdm import tqdm

from data_pipeline.config import DataPaths


def scrape_nips(year):
    nips_url = f"https://papers.nips.cc/paper/{year}"
    response = requests.get(nips_url)
    soup = BeautifulSoup(response.text, "html.parser")

    conference_items = soup.find_all('li')
    conference_items = [[ci.a.get_text(), ci.i.get_text(), ci.a['href']] for ci in conference_items]
    conference_items = [ci for ci in conference_items if ci[0]!="" and ci[1]!=""]
    return conference_items

def scrape_mlr_proceedings(conference, year):

    cy2v = {
        ("ICML", 2024): "v235",
        ("ICML", 2023): "v202",
        ("AISTATS", 2024): "v238",
        ("AISTATS", 2023): "v206",
        ("COLT", 2024): "v247",
        ("COLT", 2023): "v195",
    }

    conference_url = f"https://proceedings.mlr.press/{cy2v[(conference, year)]}"
    response = requests.get(conference_url)
    soup = BeautifulSoup(response.text, "html.parser")

    conference_items = soup.find_all('div', class_="paper")
    conference_items = [
        [
            ci.find('p', class_="title").get_text(),
            ci.find('p', class_="details").find('span', class_="authors").get_text(),
            ci.find('p', class_="links").find('a')['href']
        ]
        for ci in conference_items
    ]
    return conference_items

def scrape_aaai():
    # Scrape the technical tracks of past two years ('23, '24)
    # Look at first two pages of archives that give links to tracks
    # Look at each track

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    # First two pages
    track_links = []

    aaai_urls = [
        "https://ojs.aaai.org/index.php/AAAI/issue/archive",
        "https://ojs.aaai.org/index.php/AAAI/issue/archive/2",
    ]

    for aaai_url in aaai_urls:

        response = requests.get(aaai_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        tracks = [track.find('a', class_="title") for track in soup.find_all('h2')]
        track_links.extend(
            [(track.text.strip(), track['href']) for track in tracks if track is not None]
        )
        print(track_links)

        time.sleep(60)  # respect scraping limits
    
    # only look at past two years
    track_links = [track_link for track_link in track_links if "AAAI-24" in track_link[0] or "AAAI-23" in track_link[0]]
    print("track links: ", track_links)

    conference_items = []

    for track_link in tqdm(track_links):
        print(f"Going through track {track_link[0]} @ {track_link[1]} ")

        # Scrape tracks
        response = requests.get(track_link[1], headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        articles = soup.find_all('div', class_="obj_article_summary")

        for article in articles:

            aref = article.find('a')
            conference_items.append(
                [
                    aref.text.strip(),
                    article.find('div', class_="authors").text.strip(),
                    aref['href'],
                ]
            )

        time.sleep(60)  # respect scraping limits

    return conference_items

def scrape_emnlp(year):

    emnlp_url = f"https://{year}.emnlp.org/program/accepted_main_conference/"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    response = requests.get(emnlp_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    ps = soup.find_all('p')
    conference_items = [[p.contents[0].text, p.contents[-1].text, ''] for p in ps]
    return conference_items

def scrape_cvpr(year):
    cvpr_url = f"https://openaccess.thecvf.com/CVPR{year}?day=all"
    response = requests.get(cvpr_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Separately extract title/link and authors
    dts = soup.find_all('dt', class_="ptitle")
    conference_items = [(dt.text, '', dt.a['href']) for dt in dts]

    dds = soup.find_all('dd')
    authors = []
    for dd in dds:
        if dd.find('form') is not None:  # author entry
            authors.append(
                ', '.join([x.text for x in dd.find_all('a')])
            )

    conference_items = [[dt.text, author, dt.a['href']] for dt, author in zip(dts, authors)]
    return conference_items

def save_to_file(conference_items, filename):
    with open(filename, 'w') as f:
        for item in conference_items:
            f.write(json.dumps(item) + '\n')

def load_from_file(filename):
    with open(filename, 'r') as f:
        conference_items = [json.loads(line) for line in f]
        return conference_items

def main():

    scrape_functions = {
        "NeurIPS-2022": partial(scrape_nips, 2022),
        "NeurIPS-2023": partial(scrape_nips, 2023),
        "ICML-2023": partial(scrape_mlr_proceedings, "ICML", 2023),
        "ICML-2024": partial(scrape_mlr_proceedings, "ICML", 2024),
        "AISTATS-2023": partial(scrape_mlr_proceedings, "AISTATS", 2023),
        "AISTATS-2024": partial(scrape_mlr_proceedings, "AISTATS", 2024),
        "COLT-2023": partial(scrape_mlr_proceedings, "COLT", 2023),
        "COLT-2024": partial(scrape_mlr_proceedings, "COLT", 2024),
        "AAAI": scrape_aaai,  # easier to scrape both years at once, takes ~40 mins
        "EMNLP-2023": partial(scrape_emnlp, 2023),
        "EMNLP-2024": partial(scrape_emnlp, 2024),
        "CVPR-2023": partial(scrape_cvpr, 2023),
        "CVPR-2024": partial(scrape_cvpr, 2024),
    }

    def load_progress():
        if os.path.exists(DataPaths.CONFERENCE_DIR):
            file_paths = os.listdir(DataPaths.CONFERENCE_DIR)
            file_paths = [file_path for file_path in file_paths if file_path.endswith('.json')]
            file_paths = [file_path.split('.')[0] for file_path in file_paths]
            return set(file_paths)
        return set()

    def save_progress(conference, file_path):
        with open(file_path, 'a') as f:
            f.write(conference + '\n')

    def log_progress(msg, conference, file_path):
        with open(file_path, 'a') as f:
            f.write(conference + ': ' + msg + '\n')

    os.makedirs(DataPaths.CONFERENCE_DIR, exist_ok=True)

    # Load previous progress
    scraped_conferences = load_progress()

    # Progress file for current scrape
    progress_file = "conference_scraper_progress.tmp"

    for conference, scrape_function in tqdm(scrape_functions.items()):

        if conference in scraped_conferences:
            print(f"Skipping {conference}, already scraped.")
            log_progress("Success!", conference, progress_file)
            continue

        try:

            print(f"Scraping {conference}")
            save_path = os.path.join(DataPaths.CONFERENCE_DIR, f"{conference}.json")
            conference_items = scrape_function()
            save_to_file(conference_items, save_path)
            print(f"Saved {conference} to {str(save_path)}")
            save_progress(conference, progress_file)
            log_progress("Success!", conference, progress_file)

        except Exception as e:
            print(f"Error scraping {conference}: {e}")
            log_progress(f"ERROR: {e}", conference, progress_file)
            continue

    # Remove progress file
    os.remove(progress_file)

def stats():
    total = 0
    for fname in os.listdir(DataPaths.CONFERENCE_DIR):
        with open(os.path.join(DataPaths.CONFERENCE_DIR, fname), 'r') as file:
            num_lines = sum(1 for _ in file)
            print(fname + ": " + str(num_lines) + " lines")
            total += num_lines
    print("Total: " + str(total))


if __name__ == "__main__":
    DataPaths.ensure_directories()
    main()
    stats()
