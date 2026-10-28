from fastapi import status
from app.api.routes import HEALTH


def test_health_check(client):
    response = client.get(HEALTH)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ok"
    assert "Showroom Visitor Intelligence" in data["service"]
