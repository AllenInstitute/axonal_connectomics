import requests


def make_tiny(url):
    return requests.get(
        "http://tinyurl.com/api-create.php",
        params={"url": url}).text
