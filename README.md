# clip4ai and clip4aiLoader
Uses an OpenAI sentence transformer with the "clip-ViT-B-32" model to generate vector embeddings 
based on four images of houses. Once the data is in RAM, the embeddings are stored in DataStax Astra DB.

Standalone implementation of the [Image Search with CLIP](https://docs.datastax.com/en/astra-serverless/docs/vector-search/image-search-clip-quickstart.html) found in DataStax's documentation.

## Requirements

 - A vector-enabled [Astra DB](https://astra.datastax.com) (CQL) database;
 - its Database ID;
 - a (Database Administrator) token for the DB.
 
Copy `.env.template` to `.env`, fill the values and source it: `source .env`
(or set the same environment variables however your prefer)

Make sure you have image files in the `images` directory (there are some already).

## Functionality

### clip4aiLoader

Uses an OpenAI sentence transformer with the "clip-ViT-B-32" model to generate vector embeddings 
based on four JPG images of houses. Stores the generated embeddings in Astra DB. Creates all of the schema that it needs. Given the nature of this program, it really only needs to be run once. After that, the **clip4ai** program can be run exclusively.

### clip4ai

Requres the **clip4aiLoader** program to be run first. Queries the user for their Astra DB keyspace name. Starts with the default question, which is "a house with a swimming pool."

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
