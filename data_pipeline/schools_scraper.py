# https://medium.com/@donadviser/running-selenium-and-chrome-on-wsl2-cfabe7db4bbb

import os
import time

from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
from langchain_together import ChatTogether
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

_ = load_dotenv(find_dotenv()) # read local .env file


def get_service_and_chrome_options():
    """TODO: specific to chromedriver location."""
    # Define Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    # Add more options here if needed

    # Define paths
    user_home_dir = os.path.expanduser("~")
    user_home_dir = os.path.expanduser("~")
    chrome_binary_path = os.path.join(user_home_dir, "chrome-linux64", "chrome")
    chromedriver_path = os.path.join(user_home_dir, "chromedriver-linux64", "chromedriver")

    # Set binary location and service
    chrome_options.binary_location = chrome_binary_path
    service = Service(chromedriver_path)

    return service, chrome_options


def retrieve_csrankings_content(dump_file="soup.tmp"):
    """Write times higher page to a dump file."""
    # https://medium.com/@donadviser/running-selenium-and-chrome-on-wsl2-cfabe7db4bbb
    # Using WSL2

    service, chrome_options = get_service_and_chrome_options()

    # Initialize Chrome WebDriver
    with webdriver.Chrome(service=service, options=chrome_options) as browser:
        print("Get browser")
        browser.get("https://www.timeshighereducation.com/student/best-universities/best-universities-united-states")
        
        # Wait for the page to load
        print("Wait for the page to load")
        browser.implicitly_wait(10)

        print("Get html")
        # Retrieve the HTML content
        html_content = browser.page_source

    # Write HTML content to soup.txt
    with open(dump_file, "w") as f:
        f.write(html_content)


def extract_timeshigher_content(read_file="soup.tmp", dump_file="soup (1).tmp"):
    """Extract universities from a dump file."""
    with open(read_file, "r") as f:
        html_content = f.read()

    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Find universities
    university_table = soup.find_all('tr')
    universities = [tr.find('a').get_text() for tr in university_table if tr.find('a')]

    # Remove duplicates while keeping the order
    universities = list(dict.fromkeys(universities))

    # Write universities line-by-line to a new file
    with open(dump_file, "w") as f:
        for uni in universities:
            f.write(f"{uni}\n")


def get_department_getter():
    """
    Returns a function that leverages LangChain and TogetherAI to get a list of
    department names in a university associated with machine learning.
    """
    template_string = """\
    You are an expert in PhD programs and know about \
    specific departments at each university.\
    You are helping to design a system that generates \
    a list of professors that students interested in \
    machine learning can apply to for their PhDs. \
    Currently, recall is more important than precision. \
    Include as many departments as possible, while \
    maintaining relevancy. Which departments in {university} \
    are associated with machine learning? Please format your \
    answer as a numbered list. Afterwards, please generate a \
    new line starting with \"Answer:\", followed by a concise \
    list of department names generated, separated by
    semicolons.\
    """

    prompt_template = ChatPromptTemplate.from_template(template_string)

    # # choose from our 50+ models here: https://docs.together.ai/docs/inference-models
    chat = ChatTogether(
        together_api_key=os.environ["TOGETHER_API_KEY"],
        model="meta-llama/Llama-3-70b-chat-hf",
        temperature=0.3
    )

    output_parser = StrOutputParser()

    def extract_function(text):
        """Returns the line that starts with `Answer:`"""
        if "Answer:" not in text:
            return "No `Answer:` found"
        return text.split("Answer:")[1].strip()

    chain = prompt_template | chat | output_parser | RunnableLambda(extract_function)

    def get_department_info(uni):
        """Get department info from the university."""
        return chain.invoke({"university": uni})

    return get_department_info


def get_department_info(unis_file="soup (1).tmp", deps_file="departments.tsv"):
    """
    Get department info for all universities in `unis_file` and
    write it to `deps_file`."""

    department_getter = get_department_getter()
    with open(unis_file, "r") as fin, open(deps_file, "w") as fout:

        # Iterate through universities in `fin`
        for uni in fin.readlines():
            uni = uni.strip()

            deps = []
            # Prompt the LLM multiple times for better recall
            for i in range(3):
                depstr = department_getter(uni)
                time.sleep(3)  # Respect usage limits!
                try:
                    if depstr == "No `Answer:` found":
                        print(f"No departments found for {uni} on {i}'th prompt.")
                    else:
                        deps_ = [d.strip() for d in depstr.split(';')]
                        deps.extend(deps_)
                except Exception as e:
                    print("Exception for {uni} on {i}'th prompt: ")
                    print("Parsing string: ", depstr)
                    print(e)

            # Deduplicate deps list
            deps = list(dict.fromkeys(deps))

            # Write to tsv dump file
            for dep in deps:
                fout.write(f"{uni}\t{dep}\n")

            # Print string info
            print(f"{uni}: {deps}")
    

import requests

def get_faculty_list_potential_links_getter():
    """Returns a function that returns a list of links that may contain faculty lists."""
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    GOOGLE_SEARCH_ENGINE_ID = os.environ['GOOGLE_SEARCH_ENGINE_ID']

    def get_faculty_list_potential_links(uni, dep):
        """Returns a list of links that may contain faculty lists."""
        search_query = f'{uni} {dep} faculty list'


    params = {
        'q': search_query, 'key': GOOGLE_API_KEY, 'cx': GOOGLE_SEARCH_ENGINE_ID
    }

    response = requests.get('https://www.googleapis.com/customsearch/v1', params=params)
    results = response.json()
    title2link = {item['title']: item['link'] for item in results['items']}



# if __name__ == "__main__":
#     get_department_info()