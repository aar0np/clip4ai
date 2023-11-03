from pathlib import Path
import os
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

from multimodal_support import (
    MultiModalHuggingFaceEmbeddings,
    MultiModalCassandra,
)
import cassio


cassio.init(
    database_id=os.environ["ASTRA_DB_ID"],
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    keyspace=os.environ.get("ASTRA_DB_KEYSPACE"),
)

vector_store_name = 'images'

clip_embeddings = MultiModalHuggingFaceEmbeddings(model_name="clip-ViT-B-32")
vectorstore = MultiModalCassandra(embedding=clip_embeddings, table_name=vector_store_name, session=None, keyspace=None)

query_string = "a house with a swimming pool"

while query_string.lower() != "exit" and query_string != "":
    results = vectorstore.similarity_search(query_string, k=1)
    if results == []:
        print("\n\n** It appears that the vector store is empty. Please populate it first **\n")
        break

    result = results[0]
    image_path = result.metadata["path"]
    #
    plt.title(query_string + " / " + image_path)
    image = mpimg.imread(image_path)
    plt.imshow(image)
    plt.show()
    #
    query_string = input('Next query (empty or "exit" to leave) ? ').strip()

print("Exiting...")
