import requests

NGROK_URL = "http://0.0.0.0:8001"
API_ENDPOINT = f"{NGROK_URL.rstrip('/')}/query"

def fetch_documents(query: str, limit: int = 5, similarity_threshold: float = 0.0):
    data = {
        "query": query,
        "limit": limit,
        "similarity_threshold": similarity_threshold
    }
    try:
        response = requests.post(API_ENDPOINT, json=data, timeout=30)
        response.raise_for_status()
        raw_results = response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API request failed: {e}")
    formatted_results = []
    for item in raw_results.get("results", []):
        formatted_results.append({
            "document": item.get("pdf_name", "Unknown Document"),
            "year": item.get("year", "Unknown Year"),
            "type": item.get("doc_type", "Unknown Type"),
            "content_preview": item.get("content", "").strip()
        })
    return {
        "raw": raw_results,
        "formatted": formatted_results
    }
if __name__ == "__main__":
    results = fetch_documents("Tell me about insurance act 1938")
    print(results)
    for idx, item in enumerate(results["formatted"], start=1):
        print(f"{idx}. Document: {item['document']} ({item['year']}, {item['type']})")
        print(f"   Content Preview: {item['content_preview']}...\n")

