import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# to join the database
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facerecognition-8520d-default-rtdb.firebaseio.com/"
})

# to add data to realtime database
ref = db.reference('Customers')

data = {
    "8700107929":
        {
            "customer_name": "Abhishek Rathi",
            "customerid": "8700107929",
            "last_login_date": "2023-11-11",
            "last_login_time": "00:54:34",
            "last_login_dnt": "2022-12-11 00:54:34"
        },
    "9560071054":
        {
            "customer_name": "Tushar Ranjan",
            "customerid": "9560071054",
            "last_login_date": "2023-11-11",
            "last_login_time": "00:54:34",
            "last_login_dnt": "2022-12-11 00:54:34"
        },
    "8708724179":
        {
            "customer_name": "Rishi Varshney",
            "customerid": "8708724179",
            "last_login_date": "2023-11-11",
            "last_login_time": "00:54:34",
            "last_login_dnt": "2022-12-11 00:54:34"
        }
}

for key, value in data.items():
    ref.child(key).set(value)