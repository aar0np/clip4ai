import os
from PIL import Image
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from sentence_transformers import SentenceTransformer
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

ASTRA_DB_TOKEN_BASED_PASSWORD = os.environ['ASTRA_DB_APPLICATION_TOKEN']
ASTRA_DB_KEYSPACE = input('Your Astra Keyspace name: ')

# specify secure bundle
SECURE_CONNECT_BUNDLE_PATH = os.environ['ASTRA_SCB_PATH']

ASTRA_CLIENT_ID = 'token'
ASTRA_CLIENT_SECRET = ASTRA_DB_TOKEN_BASED_PASSWORD
KEYSPACE_NAME = ASTRA_DB_KEYSPACE
TABLE_NAME = 'images'

cloud_config= {
    'secure_connect_bundle': SECURE_CONNECT_BUNDLE_PATH
}
auth_provider = PlainTextAuthProvider(ASTRA_CLIENT_ID, ASTRA_CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider,
        protocol_version=4)
session = cluster.connect()

query_string = "a house with a swimming pool"

while query_string != "exit":
    model = SentenceTransformer('clip-ViT-B-32')
    text_emb = model.encode(query_string)

    #print(f"""
    #    Model provided embeddings for the string:
    #    '{query_string}': {text_emb.tolist()}
    #""")

    for row in session.execute(f"SELECT name, description, item_vector FROM {KEYSPACE_NAME}.{TABLE_NAME} ORDER BY item_vector ANN OF {text_emb.tolist()} LIMIT 1"):
        #print("\t" + str(row))
        plt.title(row.name)
        image = mpimg.imread(row.name)
        plt.imshow(image)
        plt.show()

    query_string = input('Next query? ')

print("Exiting...")