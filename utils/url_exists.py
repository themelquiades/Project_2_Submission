import requests

# Validates for if the remote file exists
def url_exists(url):
    if not url:
        raise ValueError("url is required")
    try:
        resp = requests.head(url)
        return True if resp.status_code == 200 else False
    except Exception as e:
        return False