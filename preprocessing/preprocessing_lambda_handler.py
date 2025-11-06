import os
import json
import boto3
import logging
from urllib.parse import unquote_plus

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sqs = boto3.client('sqs')
PREPROCESSING_QUEUE_URL = os.getenv('PREPROCESSING_QUEUE_URL')

def lambda_handler(event, context):
    """
    Triggered by S3 upload. Queues video for preprocessing.
    """
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = unquote_plus(record['s3']['object']['key'])
        
        message = {
            'video_s3_bucket': bucket,
            'video_s3_key': key
        }
        
        response = sqs.send_message(
            QueueUrl=PREPROCESSING_QUEUE_URL,
            MessageBody=json.dumps(message)
        )

        logger.info("Queued for preprocessing: s3://%s/%s", bucket, key)
        logger.info("Message ID: %s", response['MessageId'])

    return {
        'statusCode': 200,
        'body': json.dumps('Video queued for preprocessing')
    }