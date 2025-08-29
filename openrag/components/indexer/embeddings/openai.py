import base64
import requests
import openai

from pathlib import Path
from langchain_core.documents.base import Document
from openai import AsyncOpenAI, OpenAI
from utils.exceptions.embeddings import *
from utils.logger import get_logger
from config import load_config

from .base import BaseEmbedding

logger = get_logger()

config = load_config()
DATA_DIR = Path(config.paths.data_dir)

class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, embeddings_config: dict):
        self.embedding_model = embeddings_config.get("model")
        self.base_url = embeddings_config.get("base_url")
        self.api_key = embeddings_config.get("api_key")

        self._async_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        self._sync_client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    @property
    def embedding_dimension(self) -> int:
        try:
            embedding_vect = (
                self._sync_client.embeddings.create(
                    model=self.embedding_model,
                    input=["test"],
                )
                .data[0]
                .embedding
            )
            return len(embedding_vect)

        except openai.APIError as e:
            logger.error(f"API error while getting embedding dimension: {str(e)}")
            raise EmbeddingAPIError(
                f"Failed to get embedding dimension: {str(e)}",
                model_name=self.embedding_model,
                base_url=self.base_url,
                provider_error=str(e),
            )

        except (IndexError, AttributeError) as e:
            logger.error("Error while accessing embedding data", error=str(e))
            raise EmbeddingResponseError(
                "Failed to retrieve embedding dimension due to unexpected response format.",
                model_name=self.embedding_model,
                provider_error=str(e),
            )

        except Exception as e:
            logger.exception(
                "Unexpected error while getting embedding dimension", error=str(e)
            )
            raise UnexpectedEmbeddingError(
                f"Failed to get embedding dimension: {str(e)}",
                model_name=self.embedding_model,
                provider_error=str(e),
            )

    async def embed_documents(self, chunks: list[Document]) -> list[dict]:
        """
        Asynchronously embed documents using the configured embedder.
        """
        try:
            output = []
            texts = [chunk.page_content for chunk in chunks]

            for i, chunk in enumerate(chunks):
                embedding = await self._async_client.embeddings.create(
                    model=self.embedding_model,
                    input=[texts[i]],
                )
                output.append(
                    {
                        "text": chunk.page_content,
                        "data_type": "text",
                        "vector": embedding.data[0].embedding,
                        **chunk.metadata,
                    }
                )
            return output
        except openai.APIError as e:
            logger.error("API error in embed_documents", error=str(e))
            raise EmbeddingAPIError(
                f"OpenAI API error during document embedding: {str(e)}",
                model_name=self.embedding_model,
                base_url=self.base_url,
                provider_error=str(e),
            )

        except (IndexError, AttributeError) as e:
            logger.error("Error while accessing embedding data", error=str(e))
            raise EmbeddingResponseError(
                "Failed to retrieve document embeddings due to unexpected response format.",
                model_name=self.embedding_model,
                provider_error=str(e),
            )

        except Exception as e:
            logger.exception("Unexpected error while embedding documents", error=str(e))
            raise UnexpectedEmbeddingError(
                f"Failed to embed documents: {str(e)}",
                model_name=self.embedding_model,
                provider_error=str(e),
            )

    async def embed_images(self, images: list[Path], document_metatdata: dict, chunk_content_dict: dict) -> list[dict]:
        try:
            output = []
            for i, image in enumerate(images):
                image_bytes = open(str(image), "rb").read()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                data_url = f"data:image/jpeg;base64,{image_b64}"

                response = requests.post(
                    f"{self.base_url}/embeddings",
                    json={
                        "model": self.embedding_model,
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

                document_metatdata["filename"] = str(image.relative_to(DATA_DIR))
                # /home/ubuntu/an/openrag/data/pdf/charte-soja-de-france-avril2018/_page_0_Picture_0.jpeg
                output.append(
                    {
                        "text": chunk_content_dict[str(image.name)],
                        "data_type": "image",
                        "vector": response_json["data"][0]["embedding"],
                        **document_metatdata,
                    }
                )
            return output
        except openai.APIError as e:
            logger.error("API error in embed_documents", error=str(e))
            raise EmbeddingAPIError(
                f"OpenAI API error during document embedding: {str(e)}",
                model_name=self.embedding_model,
                base_url=self.base_url,
                provider_error=str(e),
            )

        except (IndexError, AttributeError) as e:
            logger.error("Error while accessing embedding data", error=str(e))
            raise EmbeddingResponseError(
                "Failed to retrieve document embeddings due to unexpected response format.",
                model_name=self.embedding_model,
                provider_error=str(e),
            )

        except Exception as e:
            logger.exception("Unexpected error while embedding documents", error=str(e))
            raise UnexpectedEmbeddingError(
                f"Failed to embed documents: {str(e)}",
                model_name=self.embedding_model,
                provider_error=str(e),
            )
        
    async def embed_query(self, query: str) -> list[float]:
        """
        Asynchronously embed a query using the configured embedder.
        """

        try:
            embedding = await self._async_client.embeddings.create(
                model=self.embedding_model,
                input=[query],
            )
            res = embedding.data[0].embedding
            return res

        except openai.APIError as e:
            logger.error("API error in embed_query", error=str(e))
            raise EmbeddingAPIError(
                f"OpenAI API error during query embedding: {str(e)}",
                model_name=self.embedding_model,
                base_url=self.base_url,
                provider_error=str(e),
            )

        except (IndexError, AttributeError) as e:
            logger.error("Error while accessing embedding data", error=str(e))
            raise EmbeddingResponseError(
                "Failed to retrieve query embedding due to unexpected response format.",
                model_name=self.embedding_model,
                provider_error=str(e),
            )

        except Exception as e:
            logger.exception("Unexpected error while embedding query", error=str(e))
            raise UnexpectedEmbeddingError(
                f"Failed to embed query: {str(e)}",
                model_name=self.embedding_model,
                provider_error=str(e),
            )
