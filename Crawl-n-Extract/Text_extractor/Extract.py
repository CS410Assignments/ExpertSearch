from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import re
import sys
import csv
import urllib
import time

links = []

def main(inputfile, outputfile):
    read_n_extract(inputfile, outputfile)

def read_n_extract(inputfile, outputfile):
    i = 0
    j = 0
    with open(inputfile,newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            link = row['url_to']
            if link in links:
                continue
            else:
                text = extract_data(link)
                if text > '':
                    if (i == 0):
                        write_text_to_file(outputfile,text)
                    else:
                        append_text_to_file(outputfile,text)
                    i = i+1

                links.append(link)

    print("total text files written = %",i)

def extract_data(link):

    text = ''
    if link is None:
        return text
    elif (is_valid_homepage(link)):
        text = extract_text_data(link)

    return text


def is_valid_homepage(link1):
    if link1 is None:
        return False
    else:
        if link1.endswith('.pdf'):  # we're not parsing pdfs
            return False

    try:
        # try to open the url
        ret_url = urllib.request.urlopen(link1).geturl()
    except:
        return False  # unable to access bio_url

    return True

def extract_text_data(link1):

    html = urlopen(link1).read()
    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()  # rip it out

    text = soup.get_text()
    text = re.sub('[^A-Za-z0-9 ]+', ' ', text)

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text1 = ' '.join(chunk for chunk in chunks if chunk)

    text2 = link1 + ' ##### ' + text1

    return text2


def write_text_to_file(filename, text):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

def append_text_to_file(filename, text):
    with open(filename, "a", encoding="utf-8") as f:
        f.write('\n')
        f.write(text)

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])