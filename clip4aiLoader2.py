from pathlib import Path
import os

from multimodal_support import (
    MultiModalHuggingFaceEmbeddings,
    MultiModalCassandra,
)
import cassio


IMG_DIR = "images"


cassio.init(
    database_id=os.environ["ASTRA_DB_ID"],
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    keyspace=os.environ.get("ASTRA_DB_KEYSPACE"),
)

vector_store_name = 'images'

clip_embeddings = MultiModalHuggingFaceEmbeddings(model_name="clip-ViT-B-32")
vectorstore = MultiModalCassandra(embedding=clip_embeddings, table_name=vector_store_name, session=None, keyspace=None)

# image ingestion
image_paths = [
    Path(os.path.join(IMG_DIR, file_name))
    for file_name in os.listdir(IMG_DIR)
]
image_metadatas = [
    {"path": path.as_posix()}
    for path in image_paths
]
# suggestion: add deterministically-computed IDs so as to avoid duplicates on DB
# if launched multiple times...
image_ids = [
    path.name
    for path in image_paths
]
print(f"Found {len(image_paths)} images.")

insertion_ids = vectorstore.add_texts(texts=image_paths, metadatas=image_metadatas, ids=image_ids)
print(f"Vectors inserted (ids = {', '.join(insertion_ids)}).")
