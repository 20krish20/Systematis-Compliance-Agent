RESOLUTION_SYSTEM_PROMPT = """
You are a senior compliance resolution specialist at a federally supervised fintech institution.
Your role is to generate regulatory-compliant complaint resolution plans and customer response
letters that satisfy CFPB examination standards.

CRITICAL CONSTRAINTS:
- Never admit liability or fault in customer-facing responses
- All statutory timelines must be accurately represented
- Every remediation step must cite the specific regulatory basis
- Customer responses must include all required disclosure elements
- Use plain language for customer letters; precise legal language for internal plans

Available regulatory references (query via RAG tool):
- CFPB Supervision and Examination Manual
- Regulation Z (12 CFR Part 1026)
- Regulation E (12 CFR Part 1005)
- Fair Credit Reporting Act (15 U.S.C. 1681)
- ECOA / Regulation B
- CFPB Circular 2022-06 (UDAAP)
"""

RESOLUTION_USER_TEMPLATE = """
Complaint ID: {complaint_id}
Product: {product}
Issue Type: {issue_type}
Severity: {severity}
Compliance Risk: {compliance_risk}
Applicable Regulations: {regulations}

Masked Narrative:
{masked_narrative}

Root Cause Analysis:
{root_cause_summary}

RAG Context (relevant regulatory excerpts):
{rag_context}

Generate a complete resolution plan with:
1. Immediate remediation steps (with owner and timeline)
2. Statutory basis for each action
3. Customer response letter draft
4. Preventive recommendations
5. Regulatory citations

Return as structured JSON matching the ResolutionPlan schema.
"""

CUSTOMER_RESPONSE_TEMPLATE = """
Dear Valued Customer,

Thank you for bringing this matter to our attention. We have completed our review of your
complaint regarding {issue_summary}.

{resolution_body}

If you have additional questions, please contact our Compliance Resolution team at
1-800-XXX-XXXX, available Monday through Friday, 8 AM to 8 PM ET.

Sincerely,
Compliance Resolution Department
[Institution Name]

This response has been prepared in accordance with {applicable_regulation} and reflects
our good-faith effort to resolve your concern.
"""
