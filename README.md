# clip4ai and clip4aiLoader
Uses an OpenAI sentence transformer with the "clip-ViT-B-32" model to generate vector embeddings 
based on four images of houses. Once the data is in RAM, the embeddings are stored in DataStax Astra DB.

Standalone implementation of the [Image Search with CLIP](https://docs.datastax.com/en/astra-serverless/docs/vector-search/image-search-clip-quickstart.html) found in DataStax's documentation.

## Requirements
 - A vector-enabled [Astra DB](https://astra.datastax.com) database
 - An Astra DB secure connect bundle
 - An Astra DB application token (with DBA priviliges)
 - Environment variables defined for: `OPENAI_API_KEY`, `ASTRA_DB_APPLICATION_TOKEN`, and `ASTRA_SCB_PATH`:

```
export ASTRA_DB_APPLICATION_TOKEN=AstraCS:GgsdfsdQuMtglFHqKZw:SDGSDDSG6a36d8526BLAHBLAHBLAHc18d40
export ASTRA_SCB_PATH=/Users/aaron.ploetz/local/secure-connect-bundle.zip
```

 - Four images of houses:

```
curl https://raw.githubusercontent.com/difli/astra-vsearch-image/main/images/one.jpg --output one.jpg
curl https://raw.githubusercontent.com/difli/astra-vsearch-image/main/images/two.jpg --output two.jpg
curl https://raw.githubusercontent.com/difli/astra-vsearch-image/main/images/three.jpg --output three.jpg
curl https://raw.githubusercontent.com/difli/astra-vsearch-image/main/images/four.jpg --output four.jpg
```

## Functionality

### clip4aiLoader
Uses an OpenAI sentence transformer with the "clip-ViT-B-32" model to generate vector embeddings 
based on four JPG images of houses. Stores the generated embeddings in Astra DB. Creates all of the schema that it needs. Given the nature of this program, it really only needs to be run once. After that, the **clip4ai** program can be run exclusively.

### clip4ai
Requres the **rclip4aiLoader** program to be run first. Queries the user for their Astra DB keyspace name. Starts with the default question, which is "a house with a swimming pool."

Once it shows the image which answers that question, (and the image window is closed) it loops to ask for more questions until the command `exit` is entered.

## Output
```
» python3 clip4ai.py

Your Astra Keyspace name: vsearch
Next query? house with a swimming pool
Next query? house with a blue sky
Next query? house with pumpkins
Next query? white house
Next query?
```