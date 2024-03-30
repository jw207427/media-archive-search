import boto3
import argparse
import os
from botocore.config import Config
import json
from tqdm import tqdm
import time

input_path =  "/opt/ml/processing/input/data"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="9bmvaevz935xeno6njp4.us-west-2.aoss.amazonaws.com")
    parser.add_argument("--movielens_index", type=str, default="mm-search-2024-03-13-19--index")
    parser.add_argument("--prefix", type=str, default="multi-modal-search")
    parser.add_argument("--bucket", type=str, default="sagemaker-us-west-2")
    parser.add_argument("--region", type=str, default="us-west-2")
    args, _ = parser.parse_known_args()

    # set env region
    os.environ['AWS_DEFAULT_REGION'] = args.region

    # import funtions
    from helper import get_text_embedding, parse_movie_data, opensearch_query
    from opensearch_util import bulk_index_ingestion

    print("check movielens data ...")
    
    try:
        entries = os.scandir(input_path)
    except FileNotFoundError as e:
        print(f"Could not scan folder: {e}")
        raise
    except OSError as e:
        print(f"Could not scan folder: {e}")
        raise

    print("process begin ...")
    
    documents = [] 

    for entry in entries:
        try:
            if entry.is_file() and entry.path.endswith((".txt")):
    
                print(f"process file: {entry.path}...")

                # Open the file in read mode using the with statement
                with open(entry.path, "r") as file:
                    # Read the contents of the file
                    contents = file.read()

                filename = os.path.basename(entry.path)
                
                doc = parse_movie_data(contents)
                doc['embedding_vector']= get_text_embedding(contents)
                doc['text']= contents
                doc['bucket']= args.bucket
                doc['key']= f"{args.prefix}/{filename}"
                
                documents.append(doc)
                
        except OSError as e: 
            print(f"Could not read entry {entry.path}: {e}")
    
    print("upload to opensearch begin ...")
    
    total_documents = len(documents)
    start_index = 0
    success_count = 0 
    failed_count = 0

    with tqdm(total=total_documents, unit="docs") as progress_bar:
        while start_index < total_documents:
            end_index = start_index + 500
            batch_documents = documents[start_index:end_index]

            success, failed = bulk_index_ingestion(args.host, args.movielens_index, batch_documents)
            time.sleep(5)
            if not isinstance(success, (list)):
                success_count += success
            if not isinstance(failed, (list)):
                failed_count += failed

            print(f"Uploading {len(batch_documents)} documents to opensearch...")

            start_index = end_index
            progress_bar.update(len(batch_documents))
    
    print(f"number of documents successfully indexed: {success_count}, number of failed ingestion: {failed_count}")

    print("validate query ...")

    # build opensearch query
    query = {
        "size": 10,
        "query": {
            "knn": {
            "embedding_vector": {
                "vector": get_text_embedding("Which movie is about toys?"),
                "k": 5
            }
            }
        },
        "_source": [ "id","bucket", "key" , "title", "genres", "text", "spoken_languages", "release_date", "rating", "cast", "overview", "budget"]
    }
    
    results = opensearch_query(query, host=args.host, index_name=args.movielens_index)

    for index, value in enumerate(results):
        print(value["_source"]["title"])

    print(f"processing complete......")