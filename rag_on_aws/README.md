# 📚 RAG on AWS

Retrieval-Augmented Generation applications built with AWS services — Amazon Bedrock, OpenSearch Serverless, S3, Textract, and more.

## Applications

### [Enterprise Knowledge Base RAG](./enterprise-knowledge-base-rag/)
Transform enterprise document repositories into a queryable knowledge base. Employees ask plain-English questions and receive accurate, cited answers powered by Amazon Bedrock (Claude), Amazon OpenSearch Serverless, and S3.

**Key AWS services:** Bedrock (Claude + Titan Embeddings), OpenSearch Serverless, S3, DynamoDB, Textract, Comprehend, SNS

**Use cases:** HR policy Q&A, legal document search, technical runbook retrieval, customer support knowledge base

### [Multi-Tenant Customer Support RAG](./customer-support-rag/)
Turn product documentation, FAQs, and resolved tickets into a conversational support assistant — with strict per-tenant vector isolation, multi-turn memory, and automatic escalation to human agents when confidence drops.

**Key AWS services:** Bedrock (Claude + Titan Embeddings), OpenSearch Serverless, S3, DynamoDB, Textract, SQS FIFO

**Use cases:** SaaS product support, e-commerce help desks, telecom device support, financial product FAQs

---

*This category is automatically updated daily.*
