import os
from PIL import Image
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from sentence_transformers import SentenceTransformer

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

session.execute(f"""CREATE TABLE IF NOT EXISTS {KEYSPACE_NAME}.{TABLE_NAME}
(id int PRIMARY KEY,
 name TEXT,
 description TEXT,
 item_vector vector<float, 512>)""")

session.execute(f"""
    CREATE CUSTOM INDEX IF NOT EXISTS image_ann_index
    ON {KEYSPACE_NAME}.{TABLE_NAME}(item_vector)
    USING 'org.apache.cassandra.index.sai.StorageAttachedIndex'
""")

session.execute(f"TRUNCATE TABLE {KEYSPACE_NAME}.{TABLE_NAME}")

model = SentenceTransformer('clip-ViT-B-32')

img_emb1 = model.encode(Image.open('one.jpg'))
img_emb2 = model.encode(Image.open('two.jpg'))
img_emb3 = model.encode(Image.open('three.jpg'))
img_emb4 = model.encode(Image.open('four.jpg'))

image_data = [
    (1, 'one.jpg', 'description1', img_emb1.tolist()),
    (2, 'two.jpg', 'description2', img_emb2.tolist()),
    (3, 'three.jpg', 'description3', img_emb3.tolist()),
    (4, 'four.jpg', 'description4', img_emb4.tolist())
]

for image in image_data:
    session.execute(f"""
        INSERT INTO {KEYSPACE_NAME}.{TABLE_NAME} (id, name, description, item_vector)
        VALUES {image}
    """)
