import logging
import boto3
from botocore.exceptions import NoCredentialsError
from pathlib import Path
import subprocess
import random
import json
import time
import requests
import cv2
import os
import base64
import re
import sagemaker
from sagemaker.utils import name_from_base
import ipywidgets as ipw
import textwrap
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

sagemaker_session = sagemaker.Session()
region = sagemaker_session._region_name

logger = logging.getLogger(__name__)


try:

    bedrock_runtime = boto3.client("bedrock-runtime")
    s3_client = boto3.client('s3')

    # initialize opensearch
    service = 'aoss'
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region, service)

except NoCredentialsError:
    print("Credentials not found. Please configure your AWS profile.")


# Using ffmpeg to extract key frames
def extract_key_frames(video_file):
    # Validate inputs
    if not os.path.exists(video_file):
        raise FileNotFoundError('Video file not found')

    # get frame rate
    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS) 

    output_dir = Path(f"/tmp/{random.randint(0, 1000000)}")
    while output_dir.exists():
        output_dir = Path(f"/tmp/{random.randint(0, 1000000)}")

    output_dir.mkdir(parents=True, exist_ok=False)

    output_file = f"{video_file.split('/')[-1].replace('.', '-')}-frame-%07d.jpg"
    output_pattern = output_dir / output_file

    # Construct ffmpeg command 
    command = ['ffmpeg', 
                '-skip_frame', 'nokey',
                '-i', video_file,
                '-vsync', 'vfr',
                '-frame_pts', 'true', 
                output_pattern]
     
    try:
        subprocess.check_output(command, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print('FFMPEG error:')
        print(err.output.decode())
        raise

    return output_dir, fps


# extract complete sentences from word transcription
def extract_sentences(items):
    sentences = []
    curr_sentence = ""
    start_time = None
    end_time = None

    try:
        for item in items:
            try:
                word = item["alternatives"][0]["content"] 
            except (KeyError, IndexError):
                logger.warning("Could not get word from %s", item)
                continue

            try:
                if start_time is None:
                    start_time = item["start_time"]
            except KeyError:
                logger.warning("Start time not found in %s", item)
                start_time = 0

            curr_sentence += word + " "

            try:
                if "end_time" in item:
                    end_time = item["end_time"]
            except KeyError:
                logger.warning("End time not found in %s", item)  

            if word in ['.','?', '!']:
                sentences.append({"sentence": curr_sentence.strip(), 
                                "start_time": start_time, 
                                "end_time": end_time})
                curr_sentence = ""
                start_time = None
    
        return sentences

    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        raise


def video_to_transcription(video_path, format="mp4"):
    transcript = []
    
    try:
        transcribe = boto3.client('transcribe')

        job_name = name_from_base(video_path.split("/")[-1].replace(".", "-"))
        job_uri = video_path

        try:
            transcribe.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': job_uri},
                MediaFormat=format,
                LanguageCode='en-US'
            )
        except Exception as e:
            logger.exception("Failed to start transcription job")
            return transcript
            
        while True:
            try:
                status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            except Exception as e:
                logger.exception("Failed to get job status")
                break

            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break
        
            print(status['TranscriptionJob']['TranscriptionJobStatus'])
            time.sleep(5)

        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            try:
                results = transcribe.get_transcription_job(TranscriptionJobName=job_name)
                output_path = results['TranscriptionJob']['Transcript']['TranscriptFileUri']
                response = requests.get(output_path)
                output = json.loads(response.text) 
                transcript = extract_sentences(output["results"]["items"])
            except Exception as e:
                logger.exception("Failed to get transcript output")
        
    except Exception as e:
        logger.exception("Unexpected error")
        
    return transcript


# check video transcript threshold
def is_within_threshold(num, start, end, threshold=0.5):
    lowest = start - threshold
    highest = end + threshold
    return lowest <= num <= highest


# extract frame and match frame to transcript
def sample_frames(video_path, transcriptions, output_dir, sample_rate):
    
    extracted = []
    try:
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)  
    except Exception as e:
        logger.error("Error opening video: %s", video_path)
        return []

    frame_gap = int(sample_rate * fps)
    
    for i in range(0, int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))):

        success, image = vidcap.read()
        if success:
            
            if i%frame_gap ==0:
                frame_time = i / fps
    
                # write the frame to file
                filename = f"frame_{i}.jpg"   
                output_path = os.path.join(output_dir, filename)  
                cv2.imwrite(output_path, image)
                
                trasncript = ""
    
                for tran in transcriptions:
                    start_time = float(tran['start_time'])
                    end_time = float(tran['end_time'])
    
                    if is_within_threshold(frame_time, 
                                           start_time, end_time, 
                                           threshold=float(sample_rate)/2):
                        trasncript += tran["sentence"] + " "
                        
                extracted.append({"video": os.path.basename(video_path),
                                  "output_file": str(output_dir / filename),
                                  "transcript": trasncript.strip(),
                                  "frame_time": frame_time})

    vidcap.release()
    return extracted


# encode frame images
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf8')


# get embeddings from bedrock
def get_embedding(image_base64=None, text_description=None):
    input_data = {}

    if image_base64 is not None:
        input_data["inputImage"] = image_base64
    if text_description is not None:
        input_data["inputText"] = text_description

    if not input_data:
        raise ValueError("At least one of image_base64 or text_description must be provided")

    body = json.dumps(input_data)

    response = bedrock_runtime.invoke_model(
        body=body,
        modelId="amazon.titan-embed-image-v1",
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get("body").read())
    return response_body.get("embedding")


# embed and then search
def embed_and_search(image_path=None, text_query=None, host="", index_name="", numb=5):

    image_base64 = None

    if image_path is not None:
        image_base64 = encode_image_to_base64(image_path)

    embedding = get_embedding(image_base64=image_base64, text_description=text_query)

    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20
    )
    # build opensearch query
    query = {
        "size": numb,
        "query": {
            "knn": {
            "multimodal_vector": {
                "vector": embedding,
                "k": 5
            }
            }
        },
        "_source": ["video", "bucket", "s3key", "timestamp"]
    }

    response = client.search(body=query, index=index_name)

    return response["hits"]["hits"]


# upload the frame images to s3 and build the index file
def upload_frames(directory, bucket, prefix, video_name, fps):
    frames = []

    try:
        for filename in os.listdir(directory):
            if filename.endswith('.jpg'): 

                # upload file to s3
                path = os.path.join(directory, filename)
                key = prefix + '/frames/' + filename
                s3_client.upload_file(path, bucket, key)
    
                # generate the time stamp
                number_str = re.search(r"-frame-(\d+)\.jpg", filename).group(1)
                number = int(number_str)

                timestamp = number*fps

                # extract embeddings
                image_base64 = encode_image_to_base64(path)
                embedding = get_embedding(image_base64=image_base64)
                
                frames.append({
                    'video': video_name, 
                    'bucket': bucket,
                    's3key': key,
                    'timestamp': timestamp,
                    'multimodal_vector': embedding,
                })
    except OSError as e:
        logging.error(f"Invalid directory: {e}")
        raise
    return frames


# render opensearch results
def render_search_result(results):
    hbox = []
    for index, value in enumerate(results):
        bucket = value["_source"]['bucket']
        key = value["_source"]['s3key']

        # retrieve image
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_content = response['Body'].read()
        image = ipw.Image(value=image_content, width=400)

        # get the metadata
        video = value["_source"]["video"]
        video_time = float(value["_source"]["timestamp"])
        description = f"From video: {video} at time: {video_time:.2f}"
        wrapped_description = "\n".join(textwrap.wrap(description, width=20))

        score = f'Score: {value["_score"]:.2%}'
        title = ipw.Label(f'{wrapped_description}\n{score}')

        hbox.append(ipw.VBox([title, image]))
    return ipw.HBox(hbox)