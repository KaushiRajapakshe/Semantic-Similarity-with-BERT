"""
    Created by KaushiRajapakshe on 27/08/2021.

    Database Logic Class
"""
# Importing all required libraries to work with the database
import firebase_admin
from firebase_admin import credentials, firestore


# Inference on database handling
class Database:

    # initialize firebase database
    def init_firebase(self):
        # Fetch the service account key JSON file contents
        cred = credentials.Certificate("ServiceAccountKey.json")
        # Initialize the app with a service account, granting admin privileges
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://bert-ss-default-rtdb.firebaseio.com/'
        })
        # Set firebase db
        self.set_db(firestore.client())
