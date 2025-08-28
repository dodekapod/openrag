import requests
import base64

image = "/home/ubuntu/an/openrag/data/pdf/Guide Synchronisation Mobile Android-1/_page_0_Picture_0.jpeg"

with open(image, "rb") as f:
    image_bytes = f.read()
image_b64 = base64.b64encode(image_bytes).decode("utf-8")
data_url = f"data:image/jpeg;base64,{image_b64}"

response = requests.post(
    "http://localhost:8000/v1/embeddings",
    json={
        "model": "jinaai/jina-embeddings-v4-vllm-retrieval",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        "encoding_format": "float",
    },
)
response.raise_for_status()
response_json = response.json()
print("Embedding output:", type(response_json["data"][0]["embedding"]))