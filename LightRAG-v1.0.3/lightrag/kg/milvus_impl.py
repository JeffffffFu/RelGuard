import asyncio
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
import numpy as np
from lightrag.utils import logger
from ..base import BaseVectorStorage

from pymilvus import MilvusClient


@dataclass
class MilvusVectorDBStorge(BaseVectorStorage):
    @staticmethod
    def create_collection_if_not_exist(
        client: MilvusClient, collection_name: str, **kwargs
    ):
        if client.has_collection(collection_name):
            return
        client.create_collection(
            collection_name, max_length=64, id_type="string", **kwargs
        )

    def __post_init__(self):
        self._client = MilvusClient(
            uri=os.environ.get(
                "MILVUS_URI",
                os.path.join(self.global_config["working_dir"], "milvus_lite.db"),
            ),
            user=os.environ.get("MILVUS_USER", ""),
            password=os.environ.get("MILVUS_PASSWORD", ""),
            token=os.environ.get("MILVUS_TOKEN", ""),
            db_name=os.environ.get("MILVUS_DB_NAME", ""),
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        MilvusVectorDBStorge.create_collection_if_not_exist(
            self._client,
            self.namespace,
            dimension=self.embedding_func.embedding_dim,
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "id": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        # asyncio.as_completed() yields in COMPLETION order while list_data below
        # is in submission order, so zipping them positionally attached whole
        # batches of vectors to the wrong records. gather() preserves order.
        embeddings_list = []
        with tqdm_async(
            total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
        ) as bar:

            async def _run(task):
                result = await task
                bar.update(1)
                return result

            embeddings_list = await asyncio.gather(
                *[_run(t) for t in embedding_tasks]
            )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["vector"] = embeddings[i]
        results = self._client.upsert(collection_name=self.namespace, data=list_data)
        return results

    async def query(self, query, top_k=5):
        embedding = await self.embedding_func([query])
        results = self._client.search(
            collection_name=self.namespace,
            data=embedding,
            limit=top_k,
            output_fields=list(self.meta_fields),
            search_params={"metric_type": "COSINE", "params": {"radius": 0.2}},
        )
        print(results)
        return [
            {**dp["entity"], "id": dp["id"], "distance": dp["distance"]}
            for dp in results[0]
        ]
