import os
import json
import hashlib
import secrets
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import base64

class SecurityUtils:
    @staticmethod
    def generate_salt():
        return secrets.token_hex(16)

    @staticmethod
    def hash_password(password, salt):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,
            backend=default_backend()
        )
        hashed = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return hashed.decode()

    @staticmethod
    def verify_password(password, salt, hashed_password):
        new_hash = SecurityUtils.hash_password(password, salt)
        return new_hash == hashed_password

    @staticmethod
    def save_credentials(credentials_path, hashed_password, salt):
        os.makedirs(os.path.dirname(credentials_path), exist_ok=True)
        with open(credentials_path, 'w', encoding='utf-8') as f:
            json.dump({
                'hashed_password': hashed_password,
                'salt': salt
            }, f)

    @staticmethod
    def load_credentials(credentials_path):
        if not os.path.exists(credentials_path):
            return None
        with open(credentials_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def get_credentials_path():
        return os.path.join(os.path.expanduser('~'), '.memoai', 'credentials.json')