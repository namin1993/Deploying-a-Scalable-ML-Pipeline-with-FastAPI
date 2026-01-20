import json
import requests

URL = "http://127.0.0.1:8000"

# ----------------------
# Send a GET request
# ----------------------
r = requests.get(URL)

# Print status code
print("GET status code:", r.status_code)

# Print welcome message
print("GET response:", r.json())


# ----------------------
# Send a POST request
# ----------------------
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

r = requests.post(
    f"{URL}/data/",
    data=json.dumps(data),
    headers={"Content-Type": "application/json"},
)

# Print status code
print("POST status code:", r.status_code)

# Print prediction result
print("POST response:", r.json())