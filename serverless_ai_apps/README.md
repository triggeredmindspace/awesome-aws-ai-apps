# Serverless AI Apps

Event-driven AI applications using **AWS Lambda**, **API Gateway**, and managed AI services — no servers to manage, scale to zero when idle.

## Applications

### [Smart Invoice Intelligence Platform](./smart-invoice-intelligence-platform/)
Automated invoice processing that extracts, validates, and categorises invoice data in real time. Eliminates manual data entry, detects anomalies, matches purchase orders, and integrates with accounting systems.

**Key AWS services:** Lambda, API Gateway, Textract, Bedrock, DynamoDB, S3, SQS

**Use cases:** Accounts payable automation, multi-vendor invoice reconciliation, spend analytics, audit trail generation

---

### [AI-Powered Product Review Intelligence Platform](./product-review-intelligence/)
Serverless platform that ingests customer product reviews via REST API or S3 batch files, analyses them with Amazon Bedrock (Claude) to extract sentiment, themes, and issues, then generates executive summaries and delivers weekly email digests.

**Key AWS services:** Lambda, API Gateway, Bedrock, SQS (FIFO), DynamoDB, S3, EventBridge, SES

**Use cases:** Product feedback aggregation, VP-level insight reports, issue prioritisation, automated weekly review digests

---

*This category is automatically updated daily.*
