import os
from PIL import Image
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from langchain.vectorstores import Cassandra
from langchain.embeddings  import HuggingFaceEmbeddings

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

embeddings = HuggingFaceEmbeddings()
vectorstore = Cassandra(embeddings, session, KEYSPACE_NAME, TABLE_NAME)

# initialize the all-MiniLM-L6-v2 model locally
model = HuggingFaceEmbeddings(model_name="clip-ViT-B-32")

# generate embeddings
img_emb1 = model.encode(Image.open('images/one.jpg'))
img_emb2 = model.encode(Image.open('images/two.jpg'))
img_emb3 = model.encode(Image.open('images/three.jpg'))
img_emb4 = model.encode(Image.open('images/four.jpg'))
img_emb5 = model.encode(Image.open('images/five.jpg'))
img_emb6 = model.encode(Image.open('images/pink_house.jpg'))

image_data = [
    (1, 'one.jpg', 'description1', img_emb1.tolist()),
    (2, 'two.jpg', 'description2', img_emb2.tolist()),
    (3, 'three.jpg', 'description3', img_emb3.tolist()),
    (4, 'four.jpg', 'description4', img_emb4.tolist()),
    (5, 'five.jpg', 'description5', img_emb5.tolist()),
    (6, 'pink_house.jpg', 'description6', img_emb6.tolist()),
]

vectorstore.add_texts(texts=image_data)
