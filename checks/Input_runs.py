import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "Hour": 2,
    "Day": 7,
    "Boundary": 0,
    "Suspicious_car_rental": 0,
    "Suspicious_fuel": 0,
    "Cumulative_type_percent": 0.5,
    "Cumulative_Unique_Locations": 5,
    "Days_since_last": 1
}

response = requests.post(url, json=data)
print(response.json())


