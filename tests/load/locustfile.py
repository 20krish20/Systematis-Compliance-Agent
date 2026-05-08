"""
Locust load test: target 1,000+ complaints/hour throughput.
Run: locust -f tests/load/locustfile.py --host=http://localhost:8000 --users=50 --spawn-rate=5
"""
from __future__ import annotations

import random

from locust import HttpUser, between, task


COMPLAINT_NARRATIVES = [
    "I have an unauthorized charge of $250 on my credit card that appeared last week. I did not make this purchase and the bank is refusing to issue a provisional credit during the dispute period as required by Regulation Z.",
    "My electronic fund transfer was reversed without notice. The bank has not provided a provisional credit within 10 business days despite my written dispute filed over 2 weeks ago. This appears to violate Regulation E.",
    "The credit bureau continues to report a debt that was discharged in bankruptcy 3 years ago. My dispute was verified without proper investigation according to the FCRA.",
    "I was denied a mortgage refinance and the loan officer mentioned my zip code as a factor. I believe this may constitute discrimination under ECOA.",
    "My student loan servicer miscalculated my income-driven repayment amount and has been overcharging me for 8 months. They refuse to provide a refund.",
    "I requested a billing dispute for a double charge in November. It has now been 95 days with no resolution, exceeding the Regulation Z 90-day maximum.",
    "My checking account was closed without notice and my direct deposit was returned. I have been unable to pay my rent as a result.",
    "The debt collector called me at work after I told them to stop. They also threatened legal action they cannot take and misrepresented the amount owed.",
    "My credit score dropped 120 points due to incorrect reporting. The bank verified the inaccurate information without contacting me.",
    "I have been charged overdraft fees on transactions that were covered by my balance at the time of processing.",
]


class ComplianceAgentUser(HttpUser):
    wait_time = between(1, 3)

    @task(70)
    def submit_complaint_async(self):
        narrative = random.choice(COMPLAINT_NARRATIVES)
        self.client.post(
            "/api/v1/complaints/submit",
            json={"narrative": narrative, "state": "CA", "submitted_via": "Web"},
            name="/api/v1/complaints/submit",
        )

    @task(20)
    def process_complaint_sync(self):
        narrative = random.choice(COMPLAINT_NARRATIVES)
        with self.client.post(
            "/api/v1/complaints/process",
            json={"narrative": narrative},
            name="/api/v1/complaints/process",
            catch_response=True,
            timeout=120,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("final_status") not in ["COMPLETED", "ESCALATED", "FAILED"]:
                    response.failure(f"Unexpected status: {data.get('final_status')}")
            elif response.status_code == 500:
                response.failure("Server error")

    @task(10)
    def health_check(self):
        self.client.get("/health", name="/health")
