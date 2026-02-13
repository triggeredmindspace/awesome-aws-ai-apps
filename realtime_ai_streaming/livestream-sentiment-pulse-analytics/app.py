```python
import asyncio
import base64
import json
import logging
import os
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Configuration ====================
class Config:
    """Application configuration from environment variables."""
    
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    KINESIS_VIDEO_STREAM = os.getenv("KINESIS_VIDEO_STREAM", "sentiment-video-stream")
    KINESIS_DATA_STREAM = os.getenv("KINESIS_DATA_STREAM", "sentiment-data-stream")
    DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "sentiment-analytics")
    DYNAMODB_EVENTS_TABLE = os.getenv("DYNAMODB_EVENTS_TABLE", "live-events")
    DYNAMODB_HIGHLIGHTS_TABLE = os.getenv("DYNAMODB_HIGHLIGHTS_TABLE", "event-highlights")
    S3_BUCKET = os.getenv("S3_BUCKET", "sentiment-analytics-data")
    BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
    ALERT_SNS_TOPIC = os.getenv("ALERT_SNS_TOPIC", "")
    CLOUDWATCH_NAMESPACE = os.getenv("CLOUDWATCH_NAMESPACE", "SentimentAnalytics")
    SENTIMENT_THRESHOLD_NEGATIVE = float(os.getenv("SENTIMENT_THRESHOLD_NEGATIVE", "-0.5"))
    SENTIMENT_THRESHOLD_POSITIVE = float(os.getenv("SENTIMENT_THRESHOLD_POSITIVE", "0.5"))
    VIRAL_MOMENT_THRESHOLD = int(os.getenv("VIRAL_MOMENT_THRESHOLD", "100"))
    API_KEY = os.getenv("API_KEY", "")


# ==================== Data Models ====================
class SentimentType(str, Enum):
    """Sentiment classification types."""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"


class EmotionType(str, Enum):
    """Emotion classification types."""
    HAPPY = "HAPPY"
    SAD = "SAD"
    ANGRY = "ANGRY"
    CONFUSED = "CONFUSED"
    DISGUSTED = "DISGUSTED"
    SURPRISED = "SURPRISED"
    CALM = "CALM"
    FEAR = "FEAR"


class StreamSource(str, Enum):
    """Supported streaming sources."""
    YOUTUBE = "YOUTUBE"
    TWITCH = "TWITCH"
    RTMP = "RTMP"
    CUSTOM = "CUSTOM"


class LiveEvent(BaseModel):
    """Live event configuration."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    stream_source: StreamSource
    stream_url: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "active"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SentimentData(BaseModel):
    """Sentiment analysis result."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_id: str
    sentiment: SentimentType
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    emotions: Dict[EmotionType, float] = Field(default_factory=dict)
    text_content: Optional[str] = None
    source_type: str = "video"  # video, audio, text, combined
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EngagementMetrics(BaseModel):
    """Real-time engagement metrics."""
    event_id: str
    timestamp: datetime
    viewer_count: int = 0
    comment_rate: float = 0.0
    reaction_count: int = 0
    average_sentiment: float = 0.0
    dominant_emotion: Optional[EmotionType] = None
    engagement_score: float = 0.0


class Alert(BaseModel):
    """Sentiment alert configuration."""
    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    event_id: str
    alert_type: str  # sentiment_shift, viral_moment, low_engagement
    severity: str  # low, medium, high, critical
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(default_factory=dict)


class Highlight(BaseModel):
    """Event highlight based on emotional peaks."""
    highlight_id: str = Field(default_factory=lambda: str(uuid4()))
    event_id: str
    start_time: datetime
    end_time: datetime
    peak_emotion: EmotionType
    peak_sentiment_score: float
    engagement_score: float
    description: str
    video_url: Optional[str] = None


# ==================== AWS Service Clients ====================
class AWSServiceManager:
    """Manages AWS service clients with proper error handling."""
    
    def __init__(self):
        """Initialize AWS service clients."""
        self.region = Config.AWS_REGION
        
        try:
            self.kinesis_video = boto3.client('kinesisvideo', region_name=self.region)
            self.kinesis_data = boto3.client('kinesis', region_name=self.region)
            self.dynamodb = boto3.resource('dynamodb', region_name=self.region)
            self.rekognition = boto3.client('rekognition', region_name=self.region)
            self.transcribe = boto3.client('transcribe', region_name=self.region)
            self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=self.region)
            self.s3 = boto3.client('s3', region_name=self.region)
            self.cloudwatch = boto3.client('cloudwatch', region_name=self.region)
            self.eventbridge = boto3.client('events', region_name=self.region)
            self.sns = boto3.client('sns', region_name=self.region)
            self.lambda_client = boto3.client('lambda', region_name=self.region)
            
            # DynamoDB tables
            self.sentiment_table = self.dynamodb.Table(Config.DYNAMODB_TABLE)
            self.events_table = self.dynamodb.Table(Config.DYNAMODB_EVENTS_TABLE)
            self.highlights_table = self.dynamodb.Table(Config.DYNAMODB_HIGHLIGHTS_TABLE)
            
            logger.info("AWS service clients initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {str(e)}")
            raise
    
    def put_cloudwatch_metric(self, metric_name: str, value: float, unit: str = "Count"):
        """Put custom metric to CloudWatch."""
        try:
            self.cloudwatch.put_metric_data(
                Namespace=Config.CLOUDWATCH_NAMESPACE,
                MetricData=[
                    {
                        'MetricName': metric_name,
                        'Value': value,
                        'Unit': unit,
                        'Timestamp': datetime.utcnow()
                    }
                ]
            )
        except ClientError as e:
            logger.error(f"Failed to put CloudWatch metric: {str(e)}")


# ==================== Sentiment Analysis Engine ====================
class SentimentAnalysisEngine:
    """Core sentiment analysis engine using AWS AI services."""
    
    def __init__(self, aws_manager: AWSServiceManager):
        """Initialize sentiment analysis engine."""
        self.aws = aws_manager
        self.sentiment_cache: Dict[str, List[SentimentData]] = {}
        
    async def analyze_video_frame(self, frame_data: bytes, event_id: str) -> Optional[SentimentData]:
        """
        Analyze video frame for facial emotions using Rekognition.
        
        Args:
            frame_data: Raw video frame bytes
            event_id: Event identifier
            
        Returns:
            SentimentData object or None if analysis fails
        """
        try:
            # Detect faces and emotions in the frame
            response = self.aws.rekognition.detect_faces(
                Image={'Bytes': frame_data},
                Attributes=['ALL']
            )
            
            if not response.get('FaceDetails'):
                return None
            
            # Aggregate emotions from all detected faces
            emotion_scores = {}
            total_confidence = 0.0
            
            for face in response['FaceDetails']:
                for emotion in face.get('Emotions', []):
                    emotion_type = emotion['Type'].upper()
                    confidence = emotion['Confidence'] / 100.0
                    
                    if emotion_type in EmotionType.__members__:
                        emotion_scores[emotion_type] = emotion_scores.get(emotion_type, 0) + confidence
                        total_confidence += confidence
            
            # Normalize emotion scores
            if total_confidence > 0:
                emotion_scores = {k: v / total_confidence for k, v in emotion_scores.items()}
            
            # Calculate overall sentiment score
            sentiment_score = self._calculate_sentiment_from_emotions(emotion_scores)
            sentiment_type = self._classify_sentiment(sentiment_score)
            
            sentiment_data = SentimentData(
                event_id=event_id,
                sentiment=sentiment_type,
                sentiment_score=sentiment_score,
                confidence=total_confidence / len(response['FaceDetails']) if response['FaceDetails'] else 0.0,
                emotions={EmotionType[k]: v for k, v in emotion_scores.items()},
                source_type="video",
                metadata={'face_count': len(response['FaceDetails'])}
            )
            
            # Store in cache
            self._cache_sentiment(event_id, sentiment_data)
            
            # Send to CloudWatch
            self.aws.put_cloudwatch_metric('VideoSentimentScore', sentiment_score)
            
            return sentiment_data
            
        except ClientError as e:
            logger.error(f"Rekognition analysis failed: {str(e)}")
            return None
    
    async def analyze_audio_sentiment(self, audio_data: bytes, event_id: str) -> Optional[SentimentData]:
        """
        Analyze audio for voice tone and transcribe text for sentiment.
        
        Args:
            audio_data: Raw audio bytes
            event_id: Event identifier
            
        Returns:
            SentimentData object or None if analysis fails
        """
        try:
            # Upload audio to S3 for Transcribe processing
            audio_key = f"audio/{event_id}/{uuid4()}.wav"
            self.aws.s3.put_object(
                Bucket=Config.S3_BUCKET,
                Key=audio_key,
                Body=audio_data
            )
            
            # Start transcription job
            job_name = f"transcribe-{event_id}-{int(time.time())}"
            self.aws.transcribe.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': f"s3://{Config.S3_BUCKET}/{audio_key}"},
                MediaFormat='wav',
                LanguageCode='en-US',
                Settings={
                    'ShowSpeakerLabels': True,
                    'MaxSpeakerLabels': 10
                }
            )
            
            # Wait for transcription (in production, use async polling)
            max_wait = 30
            waited = 0
            while waited < max_wait:
                status = self.aws.transcribe.get_transcription_job(
                    TranscriptionJobName=job_name
                )
                if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
                    break
                await asyncio.sleep(1)
                waited += 1
            
            if waited >= max_wait:
                logger.warning(f"Transcription timeout for job {job_name}")
                return None
            
            # Get transcript
            transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            # In production, fetch and parse the transcript
            transcript_text = "Sample transcript"  # Placeholder
            
            # Analyze sentiment using Bedrock
            sentiment_data = await self.analyze_text_sentiment(transcript_text, event_id)
            if sentiment_data:
                sentiment_data.source_type = "audio"
            
            return sentiment_data
            
        except ClientError as e:
            logger.error(f"Audio analysis failed: {str(e)}")
            return None
    
    async def analyze_text_sentiment(self, text: str, event_id: str) -> Optional[SentimentData]:
        """
        Analyze text sentiment using Bedrock Claude.
        
        Args:
            text: Text content to analyze
            event_id: Event identifier
            
        Returns:
            SentimentData object or None if analysis fails
        """
        try:
            # Prepare prompt for Claude
            prompt = f"""Analyze the sentiment and emotions in the following text. 
            Provide a JSON response with:
            - sentiment: POSITIVE, NEGATIVE, NEUTRAL, or MIXED
            - sentiment_score: float between -1.0 (very negative) and 1.0 (very positive)
            - confidence: float between 0.0 and 1.0
            - emotions: object with emotion types and their intensity (0.0 to 1.0)
            
            Text: {text}
            
            Respond only with valid JSON."""
            
            # Call Bedrock
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = self.aws.bedrock_runtime.invoke_model(
                modelId=Config.BEDROCK_MODEL_ID,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            content = response_body['content'][0]['text']
            
            # Parse JSON response
            analysis = json.loads(content)
            
            # Map emotions to EmotionType enum
            emotions = {}
            for emotion_name, score in analysis.get('emotions', {}).items():
                emotion_upper = emotion_name.upper()
                if emotion_upper in EmotionType.__members__:
                    emotions[EmotionType[emotion_upper]] = float(score)
            
            sentiment_data = SentimentData(
                event_id=event_id,
                sentiment=SentimentType[analysis['sentiment']],
                sentiment_score=float(analysis['sentiment_score']),
                confidence=float(analysis['confidence']),
                emotions=emotions,
                text_content=text,
                source_type="text"
            )
            
            # Store in cache
            self._cache_sentiment(event_id, sentiment_data)
            
            # Send to CloudWatch
            self.aws.put_cloudwatch_metric('TextSentimentScore', sentiment_data.sentiment_score)
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Text sentiment analysis failed: {str(e)}")
            return None
    
    async def analyze_multimodal(
        self,
        video_frame: