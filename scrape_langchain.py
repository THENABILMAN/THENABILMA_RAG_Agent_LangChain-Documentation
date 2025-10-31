import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

BASE_URL = "https://docs.langchain.com/"
visited = set()
output_file = "langchain_docs.txt"

def scrape(url):
    if url in visited: 
        return
    visited.add(url)

    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        # extract text
        text = soup.get_text(separator="\n", strip=True)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n--- PAGE: {url} ---\n{text}")

        # follow links
        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = urljoin(BASE_URL, href)

            # stay inside langchain docs domain
            if BASE_URL in full_url and full_url not in visited:
                scrape(full_url)
                time.sleep(0.2)  # be polite (avoid rate limit)

        print(f"Scraped: {url}")

    except Exception as e:
        print(f"Failed: {url} - {e}")

if __name__ == "__main__":
    print("Starting scrape...")
    scrape(BASE_URL)
    print(f"✅ Done! Docs saved to {output_file}")
