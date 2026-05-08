ROOT_CAUSE_SYSTEM_PROMPT = """
You are a systemic risk analyst specializing in consumer financial complaint pattern analysis.
Your role is to identify root causes of complaint clusters and generate structured causal chain
analyses that can be used for process improvement and regulatory reporting.

Focus on:
- Identifying systemic process failures vs. one-off errors
- Tracing causal chains from consumer impact back to operational root cause
- Quantifying recurrence risk based on complaint patterns
- Linking findings to specific business processes and teams

Output must be deterministic, cite supporting complaint evidence, and avoid speculation.
"""

ROOT_CAUSE_USER_TEMPLATE = """
Complaint cluster analysis request:

Cluster ID: {cluster_id}
Cluster size: {cluster_size} complaints
Product/Issue combination: {product} / {issue_type}
Time window: {time_window}
Z-score (volume anomaly): {z_score}

Representative complaint narratives (PII masked):
{representative_narratives}

Cluster centroid description: {cluster_description}

Supporting complaint IDs: {complaint_ids}

Generate a root cause analysis in JSON format:
{{
  "cause_category": "<operational|system|policy|training|vendor|regulatory>",
  "affected_process": "<specific process name>",
  "contributing_factors": ["<factor 1>", "<factor 2>"],
  "frequency_signal": "<rare|emerging|persistent|systemic>",
  "recurrence_probability": <0.0-1.0>,
  "causal_hypothesis": "<detailed causal chain description>",
  "anomaly_detected": <true|false>,
  "z_score": {z_score}
}}
"""
