import logging
import boto3
from botocore.config import Config
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
import numpy as np
from sagemaker.utils import name_from_base
import ipywidgets as ipw
import textwrap
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

logger = logging.getLogger(__name__)

boto_config = Config(
        connect_timeout=1, read_timeout=300,
        retries={'max_attempts': 1})

try:
    boto_session = boto3.Session()
    region = boto_session.region_name
    credentials = boto_session.get_credentials()

    bedrock_runtime = boto_session.client(
        service_name="bedrock-runtime", 
        config=boto_config
    )

    s3_client = boto_session.client('s3')

    # initialize opensearch
    service = 'aoss'
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
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vidcap.get(cv2.CAP_PROP_FPS) 

    output_dir = Path(f"/tmp/{random.randint(0, 1000000)}")
    while output_dir.exists():
        output_dir = Path(f"/tmp/{random.randint(0, 1000000)}")

    output_dir.mkdir(parents=True, exist_ok=False)

    output_file = f"{video_file.split('/')[-1].replace('.', '-')}-frame-%07d.jpg"
    output_pattern = output_dir / output_file

    if total_frames <= 1000:
        command = [
            "ffmpeg", 
            "-i", video_file,
            "-vf", "select='eq(pict_type,PICT_TYPE_I)',scale=320:240,unsharp",
            "-vsync", "vfr",
            "-f", "image2",
            output_pattern
            ]
        
        # command = f"ffmpeg -i {video_file} -vf \"select='eq(pict_type,PICT_TYPE_I)',scale=320:240,unsharp\" -vsync vfr -f image2 {output_pattern}"
    else:
        command = [
            "ffmpeg", 
            "-i", video_file,
            "-vf", "select='gt(scene,0.175)',scale=320:240,unsharp",
            "-vsync", "vfr",
            "-f", "image2",
            output_pattern
            ]
        # command = f"ffmpeg -i {video_file} -vf \"select='gt(scene,0.175)',scale=320:240,unsharp\" -vsync vfr -f image2 {output_pattern}"

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
def encode_image_to_base64(image_path, reduce=False):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    if reduce:
        # Get the original size of the image
        height, width, _ = image.shape

        # Calculate the new size (half of the original size)
        new_width = width // 2
        new_height = height // 2

        # Resize the image
        image = cv2.resize(image, (new_width, new_height))

    # Encode the resized image to base64
    _, buffer = cv2.imencode('.jpg', image)

    return base64.b64encode(buffer).decode('utf-8')


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

# get text embeddings from bedrock
def get_text_embedding(text):
    input_data = {}

    if text is None:
        raise ValueError("Test can not by None.")
    
    input_data["inputText"] = text

    body = json.dumps(input_data)

    response = bedrock_runtime.invoke_model(
        body=body,
        modelId="amazon.titan-embed-text-v1",
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get("body").read())
    return response_body.get("embedding")


# embed and then search
def opensearch_query(query, host="", index_name=""):

    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20
    )
    
    response = client.search(body=query, index=index_name)

    return response["hits"]["hits"]


# Check if a frame is empty
def is_frame_empty(frame_path, threshold_black=0.95, threshold_variation=10):

    # Load the frame image
    frame = cv2.imread(frame_path)
    
    # If the frame couldn't be loaded, return True (assuming an empty frame)
    if frame is None:
        return True
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the number of black pixels in the frame
    black_pixels = np.sum(gray < 10)
    
    # Calculate the total number of pixels in the frame
    total_pixels = gray.size
    
    # Calculate the percentage of black pixels
    black_pixel_percentage = black_pixels / total_pixels
    # Check if the percentage of black pixels exceeds the threshold
    if black_pixel_percentage > threshold_black:
        return True
    
    # Calculate the variation in pixel values
    pixel_values = np.unique(gray)
    variation = len(pixel_values)
    
    # Check if the variation in pixel values is below the threshold
    if variation < threshold_variation:
        return True
    
    return False

# upload the frame images to s3 and build the index file
def upload_frames(directory, bucket, prefix, video_name, fps):
    frames = []

    try:
        for filename in os.listdir(directory):
            if filename.endswith('.jpg'):

                # get video key
                video_key = f"{prefix}/videos/{video_name}"
                
                path = os.path.join(directory, filename)
                if not is_frame_empty(path, threshold_black=0.80, threshold_variation=220):
                    # upload file to s3
                    key = f"{prefix}/frames/{filename}"
                    s3_client.upload_file(path, bucket, key)
        
                    # generate the time stamp
                    number_str = re.search(r"-frame-(\d+)\.jpg", filename).group(1)
                    number = int(number_str)
    
                    timestamp = number // fps
    
                    # extract embeddings
                    image_base64 = encode_image_to_base64(path)
                    try:
                        embedding = get_embedding(image_base64=image_base64)
                    except Exception as e:
                        image_base64 = encode_image_to_base64(path, reduce=True)
                        embedding = get_embedding(image_base64=image_base64)
    
                    # generate frame caption
                    response = generate_caption(image_base64, text_query="What is in this image in 50 words?")
                    caption = response["content"][0]["text"]
    
                    # frame embedding ===============
                    frames.append({
                        'video': video_name, 
                        'bucket': bucket,
                        'video_key': video_key,
                        'frame_key': key,
                        'timestamp': timestamp,
                        'caption': caption,
                        'multimodal_vector': embedding,
                        'embedding_type': 'IMAGE'
                    })
                    
                    time.sleep(1)
                    
                    # caption embedding ==============
                    embedding = get_embedding(text_description=caption)
    
                    frames.append({
                        'video': video_name, 
                        'bucket': bucket,
                        'video_key': video_key,
                        'frame_key': key,
                        'timestamp': timestamp,
                        'caption': caption,
                        'multimodal_vector': embedding,
                        'embedding_type': 'TEXT'
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
        key = value["_source"]['frame_key']

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

# generate frame caption
def generate_caption(image_base64=None, text_query="What is in this image?"):

    content = []

    img_obj = dict()
    query_obj = {"type": "text", "text": text_query}
        
    if image_base64:
        img_obj["type"] = "image"
        img_obj["source"] = {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": image_base64,
        }
        content.append(img_obj)

    content.append(query_obj)

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }
    )

    response = bedrock_runtime.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=body)
    
    response_body = json.loads(response.get("body").read())

    return response_body


# parsing data from movielens dataset
def parse_movie_data(text):
    # Define regular expressions to extract key-value pairs
    id_pattern = re.compile(r"movie id: (\d+)")
    title_pattern = re.compile(r"title: (.*)")
    genres_pattern = re.compile(r"genres: (.*)")
    languages_pattern = re.compile(r"spoken languages: (.*)")
    release_date_pattern = re.compile(r"release date: (\d{4}-\d{2}-\d{2})")
    rating_pattern = re.compile(r"rating: (\d+\.\d+)")
    cast_pattern = re.compile(r"cast: (.*)")
    overview_pattern = re.compile(r"movie overview: (.*)")
    budget_pattern = re.compile(r"budget: (\d+)")

    # Initialize the movie data dictionary
    movie_data = {}

    # Extract the values using regular expressions
    match = id_pattern.search(text)
    if match:
        movie_data["id"] = int(match.group(1))

    match = title_pattern.search(text)
    if match:
        movie_data["title"] = match.group(1)

    match = genres_pattern.search(text)
    if match:
        movie_data["genres"] = match.group(1).split(", ")

    match = languages_pattern.search(text)
    if match:
        movie_data["spoken_languages"] = match.group(1).split(", ")

    match = release_date_pattern.search(text)
    if match:
        movie_data["release_date"] = match.group(1)

    match = rating_pattern.search(text)
    if match:
        movie_data["rating"] = float(match.group(1))

    match = cast_pattern.search(text)
    if match:
        movie_data["cast"] = match.group(1).split(", ")

    match = overview_pattern.search(text)
    if match:
        movie_data["overview"] = match.group(1)

    match = budget_pattern.search(text)
    if match:
        movie_data["budget"] = int(match.group(1))

    return movie_data