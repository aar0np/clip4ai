import uuid
from pathlib import Path

from typing import Any, Iterable, List, Optional, Union

from PIL import Image

from langchain.vectorstores import Cassandra

from langchain.embeddings  import HuggingFaceEmbeddings

PathOrText = Union[Path, str]

# we want to handle multi-modal, which goes beyond the text-only focus
# of the LangChain 'HuggingFaceEmbeddings' class. So, ... we subclass!
# Not very clean, langchain-wise. Probably requires a well-founded multimodal support
# (which by the looks ot if falls outside of langchain's strict scope...)
class MultiModalHuggingFaceEmbeddings(HuggingFaceEmbeddings):

    def embed_documents(self, texts: List[PathOrText]) -> List[List[float]]:
        # we split the paths and the strings, compute embeddings, then re-merge:
        # paths
        path_indices = [i for i, t in enumerate(texts) if isinstance(t, Path)]
        paths = [texts[i] for i in path_indices]
        path_embeddings = [
            self.client.encode(Image.open(path)).tolist()
            for path in paths
        ]
        # texts
        real_text_indices = [i for i, _ in enumerate(texts) if i not in path_indices]
        real_texts = [texts[i] for i in real_text_indices]
        real_text_embeddings = super().embed_documents(real_texts)
        # merge
        p_i_inv_map = {i: j for j, i in enumerate(path_indices)}
        t_i_inv_map = {i: j for j, i in enumerate(real_text_indices)}
        embeddings = [
            path_embeddings[p_i_inv_map[i]] if i in p_i_inv_map else real_text_embeddings[t_i_inv_map[i]]
            for i in range(len(texts))
        ]
        return embeddings

'''
SANITY CHECK (remove me):
    >>> v1, v2 = clip_embeddings.embed_documents([Path('images/one.jpg'), "A dragon"])
    >>> sum(x*x for x in v1)
    102.54489127209959
    >>> sum(x*x for x in v2)
    109.15439453788547
    >>> def cos(v1, v2): return sum(x*y for x,y in zip(v1,v2)) / (sum(x*x for x in v1)*sum(y*y for y in v2))**0.5
    ... 
    >>> cos(v1,v1)
    1.0
    >>> cos(v2,v2)
    1.0
    >>> cos(v1,v2)
    0.1592113019904807
    >>> v1, v2 = clip_embeddings.embed_documents([Path('images/one.jpg'), "A house with a swimming pool"])>>> cos(v1,v2)
    0.2640600993682078
    >>> v1, v2 = clip_embeddings.embed_documents([Path('images/one.jpg'), "A tall tree"])
    >>> cos(v1,v2)
    0.19165541089710356
    >>> v1, v2 = clip_embeddings.embed_documents([Path('images/one.jpg'), "A villa and the blue sky"])
    >>> cos(v1,v2)
    0.24822807719254494
'''

# Likewise, to seamlessly store image paths, we need to tweak a bit what is written
# to the database (as in: do not expect just strings to be passed...)
class MultiModalCassandra(Cassandra):

    def writeable_text(self, path_or_text: PathOrText) -> str:
        if isinstance(path_or_text, str):
            return path_or_text
        else:
            # it's a Path
            return path_or_text.as_posix()

    # This would be reworked as part of the multimodalification of the
    # LangChain structure...
    def add_texts(
        self,
        texts: Iterable[PathOrText],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 16,
        ttl_seconds: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.
            batch_size (int): Number of concurrent requests to send to the server.
            ttl_seconds (Optional[int], optional): Optional time-to-live
                for the added texts.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        _texts = list(texts)  # lest it be a generator or something
        if ids is None:
            ids = [uuid.uuid4().hex for _ in _texts]
        if metadatas is None:
            metadatas = [{} for _ in _texts]
        #
        ttl_seconds = ttl_seconds or self.ttl_seconds
        #
        embedding_vectors = self.embedding.embed_documents(_texts)
        #
        _actual_texts = [self.writeable_text(t) for t in _texts]
        for i in range(0, len(_actual_texts), batch_size):
            batch_texts = _actual_texts[i : i + batch_size]
            batch_embedding_vectors = embedding_vectors[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]

            futures = [
                self.table.put_async(
                    text, embedding_vector, text_id, metadata, ttl_seconds
                )
                for text, embedding_vector, text_id, metadata in zip(
                    batch_texts, batch_embedding_vectors, batch_ids, batch_metadatas
                )
            ]
            for future in futures:
                future.result()
        return ids
