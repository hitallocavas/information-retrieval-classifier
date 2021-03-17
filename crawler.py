import json
import requests
import time
import random


def download_html_from_url(link):
    response = requests.get(link)
    print("Baixando Link: " + link)
    if response.status_code == 200:
        return response.text


def save_json_to_file(name, content):
    with open('data/' + name + '.json', 'w') as json_file:
        json.dump(content, json_file)


def process_sambafoot():
    with open('urls/sambafoot.json') as json_file:
        urls = json.load(json_file)
        i = 18
        for url in urls:
            html = download_html_from_url(url['url'])
            content = {
                "content": html,
                "url": url['url'],
                "status": url['status']
            }
            save_json_to_file('sambafoot_html_' + str(i + 1), content)
            i += 1
            time.sleep(random.randint(60, 75))


if __name__ == '__main__':
    process_sambafoot()
