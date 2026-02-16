```python
import os
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import boto3
from botocore.exceptions import ClientError
import streamlit as st
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MeetingType(Enum):
    """Enumeration of meeting types."""
    STANDUP = "standup"
    PLANNING = "planning"
    REVIEW = "review"
    RETROSPECTIVE = "retrospective"
    ONE_ON_ONE = "one_on_one"
    ALL_HANDS = "all_hands"
    OTHER = "other"


@dataclass
class MeetingMetadata:
    """Data class for meeting metadata."""
    meeting_id: str
    title: str
    date: datetime
    duration_minutes: int
    participants: List[str]
    meeting_type: MeetingType
    organizer: str
    calendar_event_id: Optional[str] = None
    tags: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['date'] = self.date.isoformat()
        data['meeting_type'] = self.meeting_type.value
        return data


@dataclass
class ActionItem:
    """Data class for action items."""
    action_id: str
    meeting_id: str
    description: str
    assignee: str
    due_date: Optional[datetime]
    status: str
    timestamp: float
    context: str


@dataclass
class Decision:
    """Data class for decisions."""
    decision_id: str
    meeting_id: str
    description: str
    decision_maker: str
    timestamp: float
    context: str
    impact: str


class AWSConfig:
    """AWS service configuration and client management."""
    
    def __init__(self):
        """Initialize AWS clients with proper configuration."""
        self.region = os.environ.get('AWS_REGION', 'us-east-1')
        self.session = boto3.Session(region_name=self.region)
        
        # Initialize clients
        self.s3_client = self.session.client('s3')
        self.dynamodb = self.session.resource('dynamodb')
        self.lambda_client = self.session.client('lambda')
        self.stepfunctions_client = self.session.client('stepfunctions')
        self.bedrock_runtime = self.session.client('bedrock-runtime')
        self.eventbridge_client = self.session.client('events')
        
        # Configuration from environment
        self.s3_bucket = os.environ.get('MEETING_TRANSCRIPTS_BUCKET')
        self.dynamodb_table = os.environ.get('MEETING_METADATA_TABLE')
        self.opensearch_endpoint = os.environ.get('OPENSEARCH_ENDPOINT')
        self.opensearch_index = os.environ.get('OPENSEARCH_INDEX', 'meeting-transcripts')
        self.step_function_arn = os.environ.get('PROCESSING_STEP_FUNCTION_ARN')
        
        # Validate required configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate required environment variables are set."""
        required_vars = [
            'MEETING_TRANSCRIPTS_BUCKET',
            'MEETING_METADATA_TABLE',
            'OPENSEARCH_ENDPOINT'
        ]
        missing = [var for var in required_vars if not os.environ.get(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


class OpenSearchManager:
    """Manages OpenSearch operations for semantic search."""
    
    def __init__(self, config: AWSConfig):
        """Initialize OpenSearch client and vector store."""
        self.config = config
        self.session = config.session
        
        # Initialize OpenSearch client with credential provider
        self.client = OpenSearch(
            hosts=[{'host': config.opensearch_endpoint, 'port': 443}],
            http_auth=self._get_auth(),
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
        
        # Initialize embeddings
        self.embeddings = BedrockEmbeddings(
            client=config.bedrock_runtime,
            model_id="amazon.titan-embed-text-v1"
        )
        
        # Initialize vector store
        self.vector_store = OpenSearchVectorSearch(
            opensearch_url=f"https://{config.opensearch_endpoint}",
            index_name=config.opensearch_index,
            embedding_function=self.embeddings,
            http_auth=self._get_auth(),
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
    
    def _get_auth(self):
        """Get AWS authentication for OpenSearch with automatic credential refresh."""
        return AWSV4SignerAuth(
            self.session.get_credentials(),
            self.config.region,
            'es'
        )
    
    def create_index_if_not_exists(self):
        """Create OpenSearch index with proper mappings if it doesn't exist."""
        index_body = {
            "settings": {
                "index": {
                    "number_of_shards": 2,
                    "number_of_replicas": 1,
                    "knn": True
                }
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "vector_field": {
                        "type": "knn_vector",
                        "dimension": 1536
                    },
                    "meeting_id": {"type": "keyword"},
                    "timestamp": {"type": "float"},
                    "speaker": {"type": "keyword"},
                    "meeting_date": {"type": "date"},
                    "participants": {"type": "keyword"},
                    "meeting_type": {"type": "keyword"}
                }
            }
        }
        
        try:
            if not self.client.indices.exists(index=self.config.opensearch_index):
                self.client.indices.create(
                    index=self.config.opensearch_index,
                    body=index_body
                )
                logger.info(f"Created index: {self.config.opensearch_index}")
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise
    
    def index_transcript(self, meeting_id: str, transcript: str, 
                        metadata: MeetingMetadata, chunks: List[Dict[str, Any]]):
        """Index meeting transcript chunks with metadata."""
        try:
            documents = []
            for chunk in chunks:
                doc_metadata = {
                    'meeting_id': meeting_id,
                    'timestamp': chunk['timestamp'],
                    'speaker': chunk.get('speaker', 'Unknown'),
                    'meeting_date': metadata.date.isoformat(),
                    'participants': metadata.participants,
                    'meeting_type': metadata.meeting_type.value,
                    'meeting_title': metadata.title
                }
                documents.append(Document(
                    page_content=chunk['text'],
                    metadata=doc_metadata
                ))
            
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            logger.info(f"Indexed {len(documents)} chunks for meeting {meeting_id}")
            
        except Exception as e:
            logger.error(f"Error indexing transcript: {str(e)}")
            raise
    
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None, 
               k: int = 5) -> List[Document]:
        """Perform semantic search with optional filters."""
        try:
            # Build filter query
            filter_query = []
            if filters:
                if 'meeting_ids' in filters:
                    filter_query.append({
                        "terms": {"meeting_id": filters['meeting_ids']}
                    })
                if 'date_range' in filters:
                    filter_query.append({
                        "range": {
                            "meeting_date": {
                                "gte": filters['date_range']['start'],
                                "lte": filters['date_range']['end']
                            }
                        }
                    })
                if 'participants' in filters:
                    filter_query.append({
                        "terms": {"participants": filters['participants']}
                    })
                if 'meeting_type' in filters:
                    filter_query.append({
                        "term": {"meeting_type": filters['meeting_type']}
                    })
            
            # Perform search
            results = self.vector_store.similarity_search(
                query,
                k=k,
                filter=filter_query if filter_query else None
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            raise


class DynamoDBManager:
    """Manages DynamoDB operations for meeting metadata and structured data."""
    
    def __init__(self, config: AWSConfig):
        """Initialize DynamoDB manager."""
        self.config = config
        self.table = config.dynamodb.Table(config.dynamodb_table)
    
    def store_meeting_metadata(self, metadata: MeetingMetadata):
        """Store meeting metadata in DynamoDB."""
        try:
            item = metadata.to_dict()
            item['pk'] = f"MEETING#{metadata.meeting_id}"
            item['sk'] = "METADATA"
            item['gsi1pk'] = f"DATE#{metadata.date.strftime('%Y-%m-%d')}"
            item['gsi1sk'] = metadata.meeting_id
            
            self.table.put_item(Item=item)
            logger.info(f"Stored metadata for meeting {metadata.meeting_id}")
            
        except ClientError as e:
            logger.error(f"Error storing meeting metadata: {str(e)}")
            raise
    
    def store_action_items(self, action_items: List[ActionItem]):
        """Store extracted action items."""
        try:
            with self.table.batch_writer() as batch:
                for item in action_items:
                    dynamo_item = {
                        'pk': f"MEETING#{item.meeting_id}",
                        'sk': f"ACTION#{item.action_id}",
                        'gsi1pk': f"ASSIGNEE#{item.assignee}",
                        'gsi1sk': item.due_date.isoformat() if item.due_date else None,
                        'action_id': item.action_id,
                        'meeting_id': item.meeting_id,
                        'description': item.description,
                        'assignee': item.assignee,
                        'status': item.status,
                        'timestamp': item.timestamp,
                        'context': item.context
                    }
                    batch.put_item(Item=dynamo_item)
            
            logger.info(f"Stored {len(action_items)} action items")
            
        except ClientError as e:
            logger.error(f"Error storing action items: {str(e)}")
            raise
    
    def store_decisions(self, decisions: List[Decision]):
        """Store extracted decisions."""
        try:
            with self.table.batch_writer() as batch:
                for decision in decisions:
                    dynamo_item = {
                        'pk': f"MEETING#{decision.meeting_id}",
                        'sk': f"DECISION#{decision.decision_id}",
                        'gsi1pk': f"DECISION_MAKER#{decision.decision_maker}",
                        'gsi1sk': decision.meeting_id,
                        'decision_id': decision.decision_id,
                        'meeting_id': decision.meeting_id,
                        'description': decision.description,
                        'decision_maker': decision.decision_maker,
                        'timestamp': decision.timestamp,
                        'context': decision.context,
                        'impact': decision.impact
                    }
                    batch.put_item(Item=dynamo_item)
            
            logger.info(f"Stored {len(decisions)} decisions")
            
        except ClientError as e:
            logger.error(f"Error storing decisions: {str(e)}")
            raise
    
    def get_meeting_metadata(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve meeting metadata."""
        try:
            response = self.table.get_item(
                Key={
                    'pk': f"MEETING#{meeting_id}",
                    'sk': "METADATA"
                }
            )
            return response.get('Item')
            
        except ClientError as e:
            logger.error(f"Error retrieving meeting metadata: {str(e)}")
            return None
    
    def query_action_items_by_assignee(self, assignee: str, 
                                       status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query action items by assignee."""
        try:
            query_params = {
                'IndexName': 'GSI1',
                'KeyConditionExpression': 'gsi1pk = :pk',
                'ExpressionAttributeValues': {
                    ':pk': f"ASSIGNEE#{assignee}"
                }
            }
            
            if status:
                query_params['FilterExpression'] = '#status = :status'
                query_params['ExpressionAttributeNames'] = {'#status': 'status'}
                query_params['ExpressionAttributeValues'][':status'] = status
            
            response = self.table.query(**query_params)
            return response.get('Items', [])
            
        except ClientError as e:
            logger.error(f"Error querying action items: {str(e)}")
            return []
    
    def query_meetings_by_date_range(self, start_date: datetime, 
                                     end_date: datetime) -> List[Dict[str, Any]]:
        """Query meetings by date range."""
        try:
            meetings = []
            current_date = start_date
            
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                response = self.table.query(
                    IndexName='GSI1',
                    KeyConditionExpression='gsi1pk = :pk',
                    ExpressionAttributeValues={
                        ':pk': f"DATE#{date_str}"
                    }
                )
                meetings.extend(response.get('Items', []))
                current_date += timedelta(days=1)
            
            return meetings
            
        except ClientError as e:
            logger.error(f"Error querying meetings by date: {str(e)}")
            return []


class TranscriptProcessor:
    """Processes meeting transcripts and extracts structured information."""
    
    def __init__(self, config: AWSConfig):
        """Initialize transcript processor."""
        self.config = config
        self.llm = Bedrock(
            client=config.bedrock_runtime,
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            model_kwargs={
                "max_tokens": 4096,
                "temperature": 0.1
            }
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1