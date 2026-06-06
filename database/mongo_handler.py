"""
MongoDB Handler Module (OPTIONAL)
Provides database operations for storing emails and classification results.
Gracefully degrades when MongoDB is not available.
"""

import datetime

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False


class MongoHandler:
    """
    Handles MongoDB operations for the spam detection system.

    Database : spam_detection_db
    Collections:
        - emails           : raw email texts and labels
        - classifications  : prediction results with timestamps

    If MongoDB is unavailable, all methods return gracefully
    without raising exceptions, so the application can run standalone.
    """

    DB_NAME = "spam_detection_db"
    EMAILS_COLLECTION = "emails"
    CLASSIFICATIONS_COLLECTION = "classifications"

    def __init__(self, uri: str = "mongodb://localhost:27017/", timeout_ms: int = 2000):
        """
        Initialize the MongoDB connection.

        Args:
            uri: MongoDB connection URI.
            timeout_ms: Connection timeout in milliseconds.
        """
        self.client = None
        self.db = None
        self.connected = False
        self.uri = uri
        self.timeout_ms = timeout_ms

    # ── Connection ────────────────────────────────────────────────────────
    def connect(self) -> bool:
        """
        Attempt to connect to MongoDB.

        Returns:
            True if connection was successful, False otherwise.
        """
        if not PYMONGO_AVAILABLE:
            print("[MongoDB] pymongo is not installed. Running without database support.")
            return False

        try:
            self.client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=self.timeout_ms,
            )
            # Force a connection check
            self.client.admin.command("ping")
            self.db = self.client[self.DB_NAME]
            self.connected = True
            print(f"[MongoDB] Connected to {self.uri}")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError):
            print("[MongoDB] Could not connect. Running without database support.")
            self.connected = False
            return False
        except Exception as e:
            print(f"[MongoDB] Unexpected error: {e}. Running without database support.")
            self.connected = False
            return False

    # ── Emails Collection ─────────────────────────────────────────────────
    def insert_emails(self, emails: list) -> int:
        """
        Insert a list of email documents.

        Args:
            emails: List of dicts with keys 'email_text' and 'label'.

        Returns:
            Number of emails inserted, or 0 if not connected.
        """
        if not self.connected:
            return 0

        try:
            for email in emails:
                email.setdefault("inserted_at", datetime.datetime.utcnow())
            result = self.db[self.EMAILS_COLLECTION].insert_many(emails)
            print(f"[MongoDB] Inserted {len(result.inserted_ids)} emails.")
            return len(result.inserted_ids)
        except Exception as e:
            print(f"[MongoDB] Error inserting emails: {e}")
            return 0

    def get_emails(self, label: str = None, limit: int = 100) -> list:
        """
        Retrieve emails from the database.

        Args:
            label: Optional filter — 'spam' or 'ham'.
            limit: Maximum number of results.

        Returns:
            List of email documents.
        """
        if not self.connected:
            return []

        try:
            query = {"label": label} if label else {}
            cursor = self.db[self.EMAILS_COLLECTION].find(query).limit(limit)
            return list(cursor)
        except Exception as e:
            print(f"[MongoDB] Error retrieving emails: {e}")
            return []

    # ── Classifications Collection ────────────────────────────────────────
    def insert_classification(self, email_text: str, prediction: str,
                              confidence: float, preprocessed_text: str = "") -> bool:
        """
        Log a classification result.

        Args:
            email_text: The original email text.
            prediction: 'spam' or 'ham'.
            confidence: Confidence percentage (0-100).
            preprocessed_text: The preprocessed version of the email.

        Returns:
            True if inserted successfully, False otherwise.
        """
        if not self.connected:
            return False

        try:
            document = {
                "email_text": email_text,
                "prediction": prediction,
                "confidence": round(confidence, 2),
                "preprocessed_text": preprocessed_text,
                "classified_at": datetime.datetime.utcnow(),
            }
            self.db[self.CLASSIFICATIONS_COLLECTION].insert_one(document)
            return True
        except Exception as e:
            print(f"[MongoDB] Error inserting classification: {e}")
            return False

    def get_classifications(self, limit: int = 50) -> list:
        """
        Retrieve recent classification results.

        Args:
            limit: Maximum number of results.

        Returns:
            List of classification documents, newest first.
        """
        if not self.connected:
            return []

        try:
            cursor = (
                self.db[self.CLASSIFICATIONS_COLLECTION]
                .find()
                .sort("classified_at", -1)
                .limit(limit)
            )
            return list(cursor)
        except Exception as e:
            print(f"[MongoDB] Error retrieving classifications: {e}")
            return []

    # ── Cleanup ───────────────────────────────────────────────────────────
    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self.connected = False
            print("[MongoDB] Connection closed.")


# ─── Quick demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    handler = MongoHandler()
    connected = handler.connect()

    if connected:
        print("\nInserting sample emails...")
        handler.insert_emails([
            {"email_text": "You won a prize!", "label": "spam"},
            {"email_text": "Meeting at 3 PM.", "label": "ham"},
        ])

        print("\nInserting sample classification...")
        handler.insert_classification(
            email_text="Click here to win!",
            prediction="spam",
            confidence=97.5,
            preprocessed_text="click win",
        )

        print("\nRecent classifications:")
        for doc in handler.get_classifications():
            print(f"  {doc['prediction']} ({doc['confidence']}%): {doc['email_text'][:50]}")

        handler.close()
    else:
        print("MongoDB not available — app can still run without it.")
