import json
import time
import requests
import re
import random
from pathlib import Path
from typing import List, Dict
from settings.config import RAW_FILE, DATA_DIR


DATA_DIR.mkdir(exist_ok=True)
RAW_OUTPUT = RAW_FILE

MIN_ARTICLES = 500
REQUEST_DELAY = (1.5, 3.5)  
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


KEYWORDS = [
    "steel deoxidation", "non-metallic inclusions", "titanium microalloying",
    "continuous casting", "steel cleanliness", "inclusion engineering",
    "calcium treatment steel", "secondary metallurgy", "slag metal reaction",
    "aluminum killed steel", "titanium nitride inclusions", "steel refining",
    "ladle furnace", "tundish", "casting powder", "mold flux",
    "inclusion morphology", "oxide metallurgy", "clean steel production",
    "inclusion modification", "steelmaking", "continuous casting defects"
]


def extract_arxiv_year(published_date: str) -> int:
    try:
        return int(published_date[:4])
    except:
        return None

def extract_country_from_affiliation(affiliation: str) -> str:
    country_keywords = {
        'USA': ['united states', 'usa', 'u.s.', 'america'],
        'China': ['china', 'beijing', 'shanghai'],
        'Japan': ['japan', 'tokyo', 'osaka'],
        'Germany': ['germany', 'berlin', 'munich'],
        'Russia': ['russia', 'moscow', 'saint petersburg'],
        'India': ['india', 'delhi', 'mumbai'],
        'South Korea': ['south korea', 'korea', 'seoul'],
        'UK': ['united kingdom', 'uk', 'england', 'london'],
        'Sweden': ['sweden', 'stockholm'],
        'Finland': ['finland', 'helsinki'],
        'Austria': ['austria', 'vienna'],
        'Canada': ['canada', 'toronto', 'vancouver'],
        'Australia': ['australia', 'sydney', 'melbourne'],
        'Brazil': ['brazil', 'sao paulo', 'rio de janeiro']
    }
    if not affiliation:
        return "Unknown"
    affiliation_lower = affiliation.lower()
    for country, keywords in country_keywords.items():
        if any(keyword in affiliation_lower for keyword in keywords):
            return country
    return "Other"

def search_arxiv(query: str, max_results: int = 50) -> List[Dict]:
    articles = []
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f'all:"{query}"',
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    try:
        resp = requests.get(base_url, params=params, timeout=15)
        resp.raise_for_status()
        content = resp.text
        entries = re.findall(r'<entry>(.*?)</entry>', content, re.DOTALL)
        for entry in entries:
            try:
                title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                abstract = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                arxiv_id = re.search(r'<id>http://arxiv.org/abs/([^<]+)</id>', entry)
                published = re.search(r'<published>([^<]+)</published>', entry)
                authors = re.findall(r'<name>([^<]+)</name>', entry)
                if title and abstract and arxiv_id:
                    year = extract_arxiv_year(published.group(1)) if published else None
                    articles.append({
                        "title": title.group(1).strip().replace("\n", " "),
                        "abstract": abstract.group(1).strip().replace("\n", " "),
                        "source": f"arxiv:{arxiv_id.group(1)}",
                        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id.group(1)}.pdf",
                        "year": year,
                        "country": "Unknown",  # ArXiv не даёт страну
                        "authors": authors[:5],
                        "query": query
                    })
            except:
                continue
        print(f"ArXiv: найдено {len(articles)} статей")
        return articles
    except Exception as e:
        print(f"ArXiv ошибка: {e}")
        return []

def search_openalex(query: str, max_results: int = 50) -> List[Dict]:
    articles = []
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per_page": min(max_results, 50),
        "filter": "type:article",
        "select": "id,display_name,abstract_inverted_index,publication_year,authorships,primary_location,doi"
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        for work in data.get("results", []):
            abstract = ""
            if work.get("abstract_inverted_index"):
                inv_idx = work["abstract_inverted_index"]
                max_pos = max(max(positions) for positions in inv_idx.values())
                words = [""] * (max_pos + 1)
                for word, positions in inv_idx.items():
                    for pos in positions:
                        words[pos] = word
                abstract = " ".join(words)
            countries = []
            for authorship in work.get("authorships", [])[:3]:
                if authorship.get("institutions"):
                    country_code = authorship["institutions"][0].get("country_code", "")
                    country_map = {'US': 'USA', 'GB': 'UK', 'CN': 'China', 'JP': 'Japan', 'DE': 'Germany', 'RU': 'Russia'}
                    countries.append(country_map.get(country_code, country_code))
            articles.append({
                "title": work.get("display_name", ""),
                "abstract": abstract,
                "source": work.get("id", ""),
                "pdf_url": work.get("primary_location", {}).get("pdf_url", ""),
                "year": work.get("publication_year"),
                "country": countries[0] if countries else "Unknown",
                "authors": [a["author"]["display_name"] for a in work.get("authorships", [])[:5]],
                "query": query
            })
        print(f"OpenAlex: найдено {len(articles)} статей")
        return articles
    except Exception as e:
        print(f"OpenAlex ошибка: {e}")
        return []

def search_semantic_scholar(query: str, max_results: int = 30) -> List[Dict]:
    articles = []
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,year,authors,venue,url,openAccessPdf"
    }
    try:
        resp = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for paper in data.get("data", []):
            countries = []
            for author in paper.get("authors", [])[:2]:
                if author.get("affiliation"):
                    country = extract_country_from_affiliation(author["affiliation"])
                    if country != "Unknown":
                        countries.append(country)
            articles.append({
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "source": f"semanticscholar:{paper.get('paperId', '')}",
                "pdf_url": paper.get("openAccessPdf", {}).get("url", ""),
                "year": paper.get("year"),
                "country": countries[0] if countries else "Unknown",
                "authors": [a["name"] for a in paper.get("authors", [])[:3]],
                "query": query
            })
        print(f"Semantic Scholar: найдено {len(articles)} статей")
        return articles
    except Exception as e:
        print(f"Semantic Scholar ошибка: {e}")
        return []

def save_jsonl(articles: List[Dict], filepath: Path):
    seen = set()
    unique = []
    for art in articles:
        if not art.get("title") or not art.get("abstract"):
            continue
        key = f"{art['title'].lower().strip()}:{art['abstract'][:100].lower().strip()}"
        if key not in seen:
            seen.add(key)
            unique.append(art)
    with open(filepath, "a" if filepath.exists() else "w", encoding="utf-8") as f:
        for art in unique:
            f.write(json.dumps(art, ensure_ascii=False) + "\n")
    print(f"Сохранено {len(unique)} уникальных статей")

def main():
    print("Запуск парсинга металлургических статей.")
    print(f"Цель: собрать минимум {MIN_ARTICLES} статей")
    
    existing_articles = []
    if RAW_OUTPUT.exists():
        with open(RAW_OUTPUT, "r", encoding="utf-8") as f:
            existing_articles = [json.loads(line) for line in f if line.strip()]
        print(f"Уже собрано: {len(existing_articles)} статей")
    
    all_articles = existing_articles.copy()
    
    for keyword in KEYWORDS:
        if len(all_articles) >= MIN_ARTICLES * 2:
            break
        print(f"\nПоиск: '{keyword}'")
        
        # ArXiv
        arxiv = search_arxiv(keyword)
        save_jsonl(arxiv, RAW_OUTPUT)
        all_articles.extend(arxiv)
        time.sleep(random.uniform(*REQUEST_DELAY))
        
        # OpenAlex
        openalex = search_openalex(keyword)
        save_jsonl(openalex, RAW_OUTPUT)
        all_articles.extend(openalex)
        time.sleep(random.uniform(*REQUEST_DELAY))
        
        # Semantic Scholar
        ss = search_semantic_scholar(keyword)
        save_jsonl(ss, RAW_OUTPUT)
        all_articles.extend(ss)
        time.sleep(random.uniform(*REQUEST_DELAY))
        
        print(f"Текущий итог: {len(all_articles)} статей")
    
    print(f"\nСобрано: {len(all_articles)} статей")
    print(f"Файл: {RAW_OUTPUT}")

if __name__ == "__main__":
    main()