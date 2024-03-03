import boto3
import json
import time
import sagemaker
from opensearchpy.helpers import bulk
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

sagemaker_session = sagemaker.Session()
region = sagemaker_session._region_name

identity = boto3.client('sts').get_caller_identity()['Arn']

aoss_client = boto3.client('opensearchserverless')

service = 'aoss'
credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)


# bulk ingest data to opensearch index
def bulk_index_ingestion(host, index_name, data):

    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20
    )

    # Use the bulk helper to load data
    success, failed=bulk(
        client,
        data,
        index=index_name,
        raise_on_exception=True
    )

    print(f"Indexed {success} documents")

    if len(failed) > 0:
        print(f"Failed to index {failed} documents")

    return success, failed


# create an opensearch collection
def create_opensearch_collection(
    vector_store_name = "",
    index_name = "index",
    encryption_policy_name = "ep",
    network_policy_name = "np",
    access_policy_name = "ap"
):
    print(vector_store_name)
    security_policy=aoss_client.create_security_policy(
        name=encryption_policy_name,
        policy=json.dumps(
            {
                'Rules': [{'Resource': ['collection/' + vector_store_name],
                'ResourceType': 'collection'}],
                'AWSOwnedKey': True
            }),
        type='encryption'
    )

    network_policy = aoss_client.create_security_policy(
        name=network_policy_name,
        policy=json.dumps(
            [
                {'Rules': [{'Resource': ['collection/' + vector_store_name],
                'ResourceType': 'collection'}],
                'AllowFromPublic': True}
            ]),
        type='network'
    )

    collection = aoss_client.create_collection(name=vector_store_name, type='VECTORSEARCH')

    while True:
        status = aoss_client.list_collections(collectionFilters={'name': vector_store_name})['collectionSummaries'][0]['status']
        if status in ('ACTIVE', 'FAILED'): break
        time.sleep(10)

    access_policy = aoss_client.create_access_policy(
        name = access_policy_name,
        policy = json.dumps(
            [
                {
                    'Rules': [
                        {
                            'Resource': ['collection/' + vector_store_name],
                            'Permission': [
                                'aoss:CreateCollectionItems',
                                'aoss:DeleteCollectionItems',
                                'aoss:UpdateCollectionItems',
                                'aoss:DescribeCollectionItems'],
                            'ResourceType': 'collection'
                        },
                        {
                            'Resource': ['index/' + vector_store_name + '/*'],
                            'Permission': [
                                'aoss:CreateIndex',
                                'aoss:DeleteIndex',
                                'aoss:UpdateIndex',
                                'aoss:DescribeIndex',
                                'aoss:ReadDocument',
                                'aoss:WriteDocument'],
                            'ResourceType': 'index'
                        }],
                    'Principal': [identity],
                    'Description': 'Easy data policy'}
            ]),
        type='data'
    )

    return collection['createCollectionDetail']['id'] + '.' + region + '.aoss.amazonaws.com'


def create_index(os_cleint, index_name="", index_body={}):
    try:
        response = os_cleint.indices.create(index_name, body=index_body)
        print(json.dumps(response, indent=2))
    except Exception as ex:
        print(ex)
    # describe new vector index
    try:
        response = os_cleint.indices.get(index_name) 
    except Exception as ex: 
        print(ex)

    return json.dumps(response, indent=2)