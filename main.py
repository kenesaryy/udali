import asyncio
import requests
import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2 import id_token

credentials, _ = google.auth.default()
auth_token = id_token.fetch_id_token(Request(), "https://inzen-embeddings-service-756670937217.us-central1.run.app/embeddings")
credentials.refresh(Request())
auth_token = credentials.token
headers = {"Authorization": f"Bearer {auth_token}"}



def get_embeddings(prompt: str):
    response = requests.post("https://inzen-embeddings-service-756670937217.us-central1.run.app/embeddings", headers=headers, json={"text": prompt})
    print(response.json())


if __name__ == "__main__":
    get_embeddings("hello")
