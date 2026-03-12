# Bedrock AI Agents

Autonomous agents built with **Amazon Bedrock Agents** — capable of multi-step reasoning, tool use, and knowledge base retrieval without custom orchestration code.

## Applications

*Coming soon — new applications are added regularly.*

---

## What are Bedrock Agents?

Amazon Bedrock Agents let you build AI agents that:
- **Plan and execute** multi-step tasks using foundation models
- **Call APIs** via Lambda action groups (no glue code needed)
- **Query knowledge bases** backed by OpenSearch or Aurora for RAG
- **Maintain memory** across sessions with built-in DynamoDB state

### Typical architecture

```
User prompt
    │
    ▼
Bedrock Agent (Claude)
    ├─► Action Group → Lambda → external APIs / databases
    ├─► Knowledge Base → OpenSearch Serverless (RAG)
    └─► Response with citations and tool results
```

## Planned Applications

| App | Description |
|---|---|
| DevOps Automation Agent | Create tickets, query CloudWatch, restart services via chat |
| Research Synthesis Agent | Search web + internal docs, produce cited summaries |
| Data Pipeline Agent | Spin up Glue jobs, query Athena, email results |

---

*This category is automatically updated daily.*
