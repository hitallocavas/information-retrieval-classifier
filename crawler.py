import json
import requests


def download_html_from_url(link):
    response = requests.get(link)
    if response.status_code == 200:
        return response.text


def save_json_to_file(name, content):
    with open('data/' + name + '.json', 'w') as json_file:
        json.dump(content, json_file)


def process_sofascore():
    with open('urls/sofascore.json') as json_file:
        urls = json.load(json_file)
        i = 0
        for url in urls:
            html = download_html_from_url(url['url'])
            content = {
                "content": html,
                "url": url['url'],
                "status": url['status']
            }
            save_json_to_file('sofascore_html' + str(i+1), content)
            i += 1


if __name__ == '__main__':
    process_sofascore()
