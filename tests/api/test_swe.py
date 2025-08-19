from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_get_runoff_forecast():
    response = client.get("/api/v1/runoff-forecast?station_id=05OC001&start_date=2025-04-01&end_date=2025-04-07")
    assert response.status_code == 200
    data = response.json()
    assert data["station_id"] == "05OC001"
    assert "forecasts" in data
