from collections import defaultdict
import json
import os
import pickle
import requests
import time

from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from openai import OpenAI
import regex as re
from tqdm import tqdm

from data_pipeline.conference_scraper import get_authors


_ = load_dotenv(find_dotenv())

SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
SUBSCRIPTION_KEY = os.environ["BING_SEARCH_API_KEY"]
HEADERS = {
        "Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
}

EXAMPLE_PROFESSOR_JSON = {
    "is_professor": True,
    "title": "Assistant Professor",
    "department": "Computer Science", 
    "university": "Stanford University",
    "us_university": True,
}

EXAMPLE_not_professor_JSON = {
    "is_professor": False,
    "occupation": "Graduate Student",
    "affiliation": "Carnegie Mellon University"
}

IS_PROFESSOR_TEMPLATE = """You are a helpful assistant tasked with determining if {person_name} is a machine learning \
professor. You have search results from the query "{person_name} machine learning professor". \
Based on the results, specify if {person_name} is a professor, and if so, provide \
their title, department, university, and whether their university is in the U.S. If not, give their occupation \
and affiliation. Note: multiple people may \
share the same name, so choose the one most likely in machine learning. Further, one person may have multiple \
positions. If this is the case and one of those positions include being a professor, specify they are a professor \
and provide their title, department, university, and whether their university is in the U.S.

Only return the raw JSON, no MarkDown!

If {person_name} **is** a professor, fill out:
- `is_professor`: true
- `title`: e.g., `Assistant Professor`, `Associate Professor`, `Professor` etc.
- `department`: Full name, e.g., `Computer Science` rather than `CS` and `Electrical Engineering` rather than `EE`.
- `university`: Full name, e.g., `California Institute of Technology` rather than `Caltech`
- `us_university`: `true` or `false`

Example:
{professor_json_template}

If {person_name} **is not** a professor, fill out:
- `is_professor`: false
- `occupation`: e.g., `Graduate Student`, `Researcher`, `Engineer`, `Scientist`
- `affiliation`: e.g., `Carnegie Mellon University`, `Deepmind`, `Apple`, `NVIDIA`

Example:
{not_professor_json_template}

Search results (formatted as a numbered list with link name and snippet). \
Again, only return the JSON, just with the dictionary and its fields.
{hits}"""

# import httpx
def bing_search(person_name, max_retries=0, wait_time=0.5):
    """Performs the bing search `person_name` machine learning professor."""
    query = "{} machine learning professor".format(person_name)
    params = {"q": query, "count": 10, "offset": 0, "mkt": "en-US", "textFormat": "HTML"}

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(SEARCH_URL, headers=HEADERS, params=params)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as http_err:
            if attempt == max_retries:
                raise Exception(f"Max retries reached. Failed to get a valid response for {person_name}") from http_err
            print(f"An error occurred while searching {person_name}: {http_err}. Retrying in {wait_time} seconds ...")
            time.sleep(wait_time)

    return ""  # doesn't run

def process_search_results(search_results):
    """Cleans up bing search results."""
    # What people see, url name and snippet
    readable_results = "\n".join(["{0}. [{1}]: [{2}]".format(i + 1, v["name"], v["snippet"])
                                    for i, v in enumerate(search_results["webPages"]["value"])])
    soup = BeautifulSoup(readable_results, "html.parser")
    cleaned_readable_results = soup.get_text()
    cleaned_readable_results = re.sub(r'[^\x00-\x7F]+', '', cleaned_readable_results)

    # Links
    url_results = "\n".join(["{0}. {1}".format(i + 1, v["url"])
                                    for i, v in enumerate(search_results["webPages"]["value"])])

    # Combine human readable and links
    web_results = [cleaned_readable_results, url_results]
    return web_results

def get_prompt(person_name, top_hits):
    template = PromptTemplate(
        input_variables=["person_name", "professor_json_template", "not_professor_json_template", "hits"],
        template=IS_PROFESSOR_TEMPLATE,
    )

    filled_prompt = template.format(person_name=person_name,
                        professor_json_template=json.dumps(EXAMPLE_PROFESSOR_JSON),
                        not_professor_json_template=json.dumps(EXAMPLE_not_professor_JSON),
                        hits="\n".join(top_hits))

    return filled_prompt

def run_chatgpt(prompt, client, model="gpt-4o-mini", system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        seed=123,
    )

    # Return response
    return response.choices[0].message.content

def check_json(profile):
    if not isinstance(profile, dict):
        raise ValueError("Profile must be a dictionary")
    
    if "is_professor" not in profile:
        raise ValueError("Profile must contain a 'is_professor' key")

    if profile["is_professor"]:
        if "title" not in profile:
            raise ValueError("Profile must contain a 'title' key")
        if "department" not in profile:
            raise ValueError("Profile must contain a 'department' key")
        if "university" not in profile:
            raise ValueError("Profile must contain a 'university' key")
        if "us_university" not in profile:
            raise ValueError("Profile must contain a 'us_university' key")
        if type(profile["us_university"]) is not bool:
            raise ValueError("Profile 'us_university' must be a boolean")
    else:
        if "occupation" not in profile:
            raise ValueError("Profile must contain an 'occupation' key")
        if "affiliation" not in profile:
            raise ValueError("Profile must contain an 'affiliation' key")

def save_json(profiles, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:  # appending just the new ones would be better
        json.dump(profiles, file, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def log_progress_to_file(progress_log, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as file:
        file.write('\n'.join(progress_log))
        file.write('\n' + '-' * 10 + '\n')

def search_person(person_name, progress_log):
    """Completes a bing search for the person."""
    try:
        search_results = bing_search(person_name)
        web_results = process_search_results(search_results)
        top_hits = web_results[0].split("\n")[:5]  # Extract top 5 results
        progress_log.append(f"Success: Search results for {person_name}.")
        return top_hits
    except Exception as e:
        print(f"Search exception for {person_name}: ", e)
        progress_log.append(f"Failure: Search exception for {person_name}: {e}")
        return ""
    
def extract_search_results(person_name, progress_log, client, us_professor_profiles, not_us_professor_profiles, top_hits):
    """Use LLM to extract data from search results."""
    try:
        prompt = get_prompt(person_name, top_hits)
        gpt_output = run_chatgpt(prompt, client)  # LLM plz help
        gpt_json = json.loads(gpt_output)
        gpt_profile = {"name": person_name}
        gpt_profile.update(gpt_json)
        check_json(gpt_profile)
        if gpt_profile["is_professor"] and gpt_profile["us_university"]:
            us_professor_profiles.append(gpt_profile)
        else:
            not_us_professor_profiles.append(gpt_profile)
    except Exception as e:
        print(f"LLM exception for {person_name}: ", e)
        progress_log.append(f"Failure: LLM exception for {person_name}: {e}")

def research_person(person_name, client, progress_log, us_professor_profiles, not_us_professor_profiles):
    """Research who this person is and save results."""
    top_hits = search_person(person_name, progress_log)
    if top_hits == "":
        return
    extract_search_results(person_name, progress_log, client, us_professor_profiles, not_us_professor_profiles, top_hits)


def get_authors(save_dir="data/conference", min_papers=3, ignore_first_author=True):
    """
    Reduce the list of authors to those with at least `min_papers` papers for
    which they are not first authors. Ignores solo-authored papers and papers
    with more than 20 authors.

    Filters authors so that we don't have to do RAG on every author, which is
    monetarily expensive. Feel free to edit if you have more resources.
    """
    authors = defaultdict(int)
    for fname in os.listdir(save_dir):
        if not fname.endswith('.json'):
            continue

        with open(os.path.join(save_dir, fname), 'r') as file:
            for line in file:
                item = json.loads(line)
                paper_authors = [x.strip() for x in item[1].split(",")]

                # ignore solo-authored papers and papers with more than 20 authors
                if len(paper_authors) == 1 or len(paper_authors) > 20:
                    continue

                # professors generally are not first authors
                if not ignore_first_author and len(paper_authors) > 0:
                    authors[paper_authors[0]] += 1
                for i in range(1, len(paper_authors)):
                    authors[paper_authors[i]] += 1

    authors = {k: v for k, v in authors.items() if v >= min_papers}
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "authors.txt"), 'w') as f:
        for k, v in authors.items():
            f.write(f"{k}\t{v}\n")
    return authors

def research_conference_profiles(save_freq=20):
    """Research each author as a stream.
    
    NOTE: cannot deal w/ interrupts and continue from past progress.
    """

    authors = get_authors("data/conference")
    person_names = list(authors.keys())

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    progress_log = []
    us_professor_profiles = []
    not_us_professor_profiles = []

    def log_save_print(progress_log, us_professor_profiles, not_us_professor_profiles, i):
        log_progress_to_file(progress_log, 'logs/progress_log.tmp')
        save_json(us_professor_profiles, 'data/professor/us_professor.json')
        save_json(not_us_professor_profiles, 'data/professor/not_us_professor.json')
        print(f"Saved profiles to data/professor/us_professor.json and data/professor/not_us_professor.json after processing {i} people")

    for i in range(len(person_names)):
        research_person(person_names[i], client, progress_log, us_professor_profiles, not_us_professor_profiles)
        if i % save_freq == 0:
            log_save_print(progress_log, us_professor_profiles, not_us_professor_profiles, i)

    log_save_print(progress_log, us_professor_profiles, not_us_professor_profiles, i)
    print("Research complete.")

def batch_search_person(person_names, progress_log, save_freq=20):
    """Searches everyone given in `person_names`."""
    # might start and stop, pull from previous efforts
    try:
        prev_researched_authors = load_json("data/professor/search_results.json")
    except:
        prev_researched_authors = []
    ignore_set = set([x[0] for x in prev_researched_authors])
    data = prev_researched_authors
    unseen_person_names = []
    for person_name in person_names:
        if person_name not in ignore_set:
            unseen_person_names.append(person_name)
    print(f"Already researched {len(ignore_set)} / {len(person_names)} = {len(ignore_set) / len(person_names)} of the dataset")
    person_names = unseen_person_names

    # continue search
    for i in tqdm(range(len(person_names))):
        if person_names[i] in ignore_set:
            continue  # seen before

        query_start = time.time()
        top_hits = search_person(person_names[i], progress_log)
        if top_hits != "":
            data.append([person_names[i], top_hits])

        if i % save_freq == 0:
            save_json(data, "data/professor/search_results.json")
            log_progress_to_file(progress_log, 'logs/progress_log.tmp')

        # 3 queries per second max
        wait_time = max(time.time() - (query_start + 0.334), 0.0)
        time.sleep(wait_time)

    save_json(data, "data/professor/search_results.json")
    log_progress_to_file(progress_log, 'logs/progress_log.tmp')

def write_batch_files(search_results_path,
                      prompt_data_path_prefix,
                      model="gpt-4o-mini",
                      max_tokens=1000,
                      temperature=0.0,
                      seed=123,
                      batch_size=1999,  # max_tokens * batch_size < 2M?
                      verbose=0):
    """Convert search results dump to jsonl for LLM batch request."""
    with open(search_results_path, "r") as f:
        search_results = json.load(f)

    prompt_datas = []
    for search_result in search_results:
        prompt_data = {
            "custom_id": f"request-{search_result[0]}",  # don't change, needed for decoding
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "messages": [{"role": "user", "content": get_prompt(search_result[0], search_result[1])}],
                "max_tokens": max_tokens
            }
        }
        prompt_datas.append(prompt_data)
    
    print(f"Number of prompts: {len(prompt_datas)}")
    if verbose > 0:
        print(get_prompt(search_result[0], search_result[1]))

    batch_paths = []
    for i in range(0, len(prompt_datas) // batch_size + 1):
        prompt_data_path = f"{prompt_data_path_prefix}_{i}.jsonl"
        batch_range = i * batch_size, (min(len(prompt_datas), (i + 1) * batch_size))
        with open(prompt_data_path, "w") as f:
            for prompt_data in prompt_datas[batch_range[0]:batch_range[1]]:
                f.write(json.dumps(prompt_data) + "\n")
        batch_paths.append(prompt_data_path)

    return batch_paths

def send_batch_files(prompt_data_path_prefix, batch_paths, client, timeout=24*60*60):
    """Create and send the batch request to API endpoint."""
    batches = []

    print("Batching and sending requests...")
    for batch_path in tqdm(batch_paths):
        batch_input_file = client.files.create(
            file=open(batch_path, "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id
        print(f"Batch input file ID: {batch_input_file_id}")

        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": "search extraction job"
            }
        )

        begin = time.time()
        while time.time() - begin < timeout:
            batch = client.batches.retrieve(batch.id)
            if batch.status == "completed":
                break
            time.sleep(40)
            print(f"Status ({time.time()-begin:2f}): {batch.status}")
            print("seconds elapsed: ", time.time() - begin)
        batches.append(batch)

    # Keeps track of the paths to the batch files
    with open(f"{prompt_data_path_prefix}_batches.pkl", "wb") as f:
        pickle.dump(batches, f)
    with open(f"{prompt_data_path_prefix}_ids.txt", "w") as f:
        f.write("\n".join([x.id for x in batches]))
    return batches

def retrieve_batch_output(client, batch_id):
    """OpenAI batch requests finish within 24 hrs."""
    retrieved_batch = client.batches.retrieve(batch_id)
    if retrieved_batch.status == "completed":
        return client.files.content(retrieved_batch.output_file_id).text
    else:
        print("Batch process is still in progress.")
        print(retrieved_batch)
        return "INCOMPLETE"

def batch_process_llm_output(client, batches):
    client = OpenAI()

    outputs = []
    for batch in batches:
        batch_id = batch.id
        output = retrieve_batch_output(client, batch_id)
        if output == "INCOMPLETE":
            return
        outputs.append(output)

    for output in outputs:
        json_objects = output.split('\n')
        custom_id_idx = len("request-")  # where the name begins in "custom_id"

        progress_log = []
        us_professor_profiles = []
        not_us_professor_profiles = []

        for json_obj in json_objects:
            if json_obj == '': continue

            try:
                parsed_data = json.loads(json_obj)
                message_content = parsed_data["response"]["body"]["choices"][0]["message"]["content"]
                gpt_json = json.loads(message_content)
                gpt_profile = {"name": parsed_data["custom_id"][custom_id_idx:]}
                gpt_profile.update(gpt_json)
                check_json(gpt_profile)
                if gpt_profile["is_professor"] and gpt_profile["us_university"]:
                    us_professor_profiles.append(gpt_profile)
                else:
                    not_us_professor_profiles.append(gpt_profile)

                progress_log.append(f"Success: Parsed LLM output for {gpt_profile['name']}")
            except Exception as e:
                try:
                    print(f"Failed to parse json object for custom-id `{parsed_data['custom_id']}`: {e}")
                    progress_log.append(f"Failed: Parsed LLM output for {gpt_profile['name']}: {e}")
                except Exception as e2:
                    print(f"Failed to parse json object `{json_obj}`: {e2}")
                    progress_log.append(f"Failed UNKNOWN: Parsed LLM output: {e2}")

    with open("data/professor/us_professor.json", 'w') as file:
        json.dump(us_professor_profiles, file, indent=4)
    with open("data/professor/not_us_professor.json", 'w') as file:
        json.dump(not_us_professor_profiles, file, indent=4)

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="US Professor Verifier: Search or LLM-Analyze batch operations."
    )
    
    # Add mutually exclusive group to ensure only one of the arguments is passed
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--batch_search',
        action='store_true',
        help='Batch search the authors.'
    )
    group.add_argument(
        '--batch_analyze',
        action='store_true',
        help='Sends search results to LLM for analysis.'
    )
    group.add_argument(
        '--batch_retrieve',
        action='store_true',
        help='Retrieve results from an LLM batch request, requires --batch_id'
    )

    parser.add_argument(
        '--batch_ids_path',
        type=str,
        help='The batch ID for retrieval'
    )

    args = parser.parse_args()

    prompt_data_path_prefix = "data/professor/prompt_data"

    if args.batch_search:
        authors = get_authors("data/conference")
        authors_list = list(authors.keys())
        print("Researching people...")
        progress_log = []
        batch_search_person(authors_list, progress_log, save_freq=20)
    elif args.batch_analyze:
        client = OpenAI()
        batch_paths = write_batch_files("data/professor/search_results.json", prompt_data_path_prefix)
        send_batch_files(prompt_data_path_prefix, batch_paths, client)
    elif args.batch_retrieve:
        client = OpenAI()
        with open(f"{prompt_data_path_prefix}_batches.pkl", "rb") as f:
            batches = pickle.load(f)
        batch_process_llm_output(client, batches)
    else:
        raise ValueError("Please specify --batch_search, --batch_analyze, or --batch_retrieve.")


if __name__ == "__main__":
    main()
