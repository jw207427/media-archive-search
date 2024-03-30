import pandas as pd
import argparse
import pathlib
import json
import os
import numpy as np

input_path =  "/opt/ml/processing/input/videos"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="9bmvaevz935xeno6njp4.us-west-2.aoss.amazonaws.com")
    parser.add_argument("--index_name", type=str, default="mm-search-2024-03-13-19--index")
    parser.add_argument("--prefix", type=str, default="multi-modal-search")
    parser.add_argument("--bucket", type=str, default="sagemaker-us-west-2")
    parser.add_argument("--region", type=str, default="us-west-2")
    args, _ = parser.parse_known_args()

    # set env region
    os.environ['AWS_DEFAULT_REGION'] = args.region

    # import funtions
    from helper import extract_key_frames, upload_frames, get_embedding, opensearch_query
    from opensearch_util import bulk_index_ingestion
    
    print("check video files ...")
    
    try:
        entries = os.scandir(input_path)
    except FileNotFoundError as e:
        print(f"Could not scan folder: {e}")
        raise
    except OSError as e:
        print(f"Could not scan folder: {e}")
        raise

    print("process begin ...")
    
    for entry in entries:
        try:
            if entry.is_file() and entry.path.endswith((".mp4", ".mkv", ".mov")):
    
                print(f"process file: {entry.path}...")
                # extract key frames from video
                output_dir, fps = extract_key_frames(entry.path)
                print(f"frame rate is {fps} per second")
    
                print(f"number of frames: {len(os.listdir(output_dir))}")
                # upload fames to s3
                frames = upload_frames(output_dir, args.bucket, args.prefix, entry.path.split('/')[-1], fps)
                
                print(f"upload frames to s3 bucket...")
    
                # ingest the index file into opensearch
                sucess, failed = bulk_index_ingestion(args.host, args.index_name, frames)

                print(f"{sucess} record succeeded, {failed} failed ...")
                
        except OSError as e: 
            print(f"Could not read entry {entry.path}: {e}")

    print("validate query ...")

    # build opensearch query
    query = {
        "size": 3,
        "query":{
            "knn": {
            "multimodal_vector": {
                "vector": get_embedding(text_description="yellow cars"),
                "k": 5
            }
            }
        },
        "_source": ["video", 
                    "bucket", 
                    "video_key" , 
                    "frame_key", 
                    "caption", 
                    "timestamp", 
                    "embedding_type"]
    
        }
    
    results = opensearch_query(query,
                               host = args.host,
                               index_name=args.index_name)

    for index, value in enumerate(results):
        print(value["_source"]["caption"])

    print(f"processing complete......")