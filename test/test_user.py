import pytest

@pytest.mark.parametrize("email, password, status_code", [
    ("valid_email@gmail.com", "password123", 201),
    ("invalid_email#gmail.com", "password123", 422),
    ("valid_email@gmail.com", "qwe", 400),
    ("valid_email@gmail.com", None, 422),
    (None, "secretpassword", 422)
])
def test_create_user(client, email, password, status_code):
    res = client.post(
        "/users/", json={"email":email , "password": password})

    assert res.status_code == status_code

