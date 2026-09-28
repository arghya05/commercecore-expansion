"""Locust load test for the expansion serving API. Run headless:
locust -f expansion/serve/load_test.py --host http://127.0.0.1:8899 \
    --headless -u 5 -r 1 -t 60s --csv reports/load_test_2026-09-28/result

Local CPU inference is slow (no GPU on this machine) -- this test measures
server behavior under concurrent load (queuing, error rate, memory), not
production latency. Real per-request latency was already measured on GPU
elsewhere (understand adapter: ~100ms/request per earlier serving fix).
"""
from locust import HttpUser, task, between


class ExpansionApiUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(2)
    def match_relevance(self):
        self.client.post("/v1/match/relevance", json={
            "tenant_id": "loadtest",
            "query": "wireless bluetooth headphones",
            "product_title": "Sony WH-1000XM5 Wireless Noise Canceling Headphones",
        })

    @task(2)
    def match_identity(self):
        self.client.post("/v1/match/identity", json={
            "tenant_id": "loadtest",
            "title_a": "Apple iPhone 15 128GB Blue",
            "title_b": "iPhone 15, 128GB, Blue - Unlocked",
        })

    @task(1)
    def catalog_normalize(self):
        self.client.post("/v1/catalog/normalize", json={
            "tenant_id": "loadtest",
            "text": "Nike Air Max 270 Women's Trainers - Black/White. Breathable mesh upper.",
        })

    @task(1)
    def health(self):
        self.client.get("/health")
