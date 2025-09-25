from passlib.hash import argon2
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

def hash_password(plain):
    return argon2.using(type="argon2id", memory_cost=2**16, time_cost=3).hash(plain)

def verify_password(hash, plain):
    return argon2.verify(plain, hash)

def generate_data_key():
    """
    Génère une clé symétrique AES-256 pour chiffrer un fichier
    """
    return AESGCM.generate_key(bit_length=256)

def encrypt_file(input_path, output_path, data_key):
    """
    Chiffre un fichier avec AES-GCM et stocke IV + ciphertext
    """
    with open(input_path, "rb") as f:
        plaintext = f.read()
    
    aesgcm = AESGCM(data_key)
    iv = os.urandom(12)  # nonce unique
    ciphertext = aesgcm.encrypt(iv, plaintext, None)
    
    with open(output_path, "wb") as f:
        f.write(iv + ciphertext)
    print(f"Fichier chiffré créé : {output_path}")

def decrypt_file(input_path, output_path, data_key):
    """
    Déchiffre un fichier AES-GCM à partir de IV + ciphertext
    """
    with open(input_path, "rb") as f:
        data = f.read()
    
    iv = data[:12]
    ciphertext = data[12:]
    
    aesgcm = AESGCM(data_key)
    plaintext = aesgcm.decrypt(iv, ciphertext, None)
    
    with open(output_path, "wb") as f:
        f.write(plaintext)
    print(f"Fichier déchiffré créé : {output_path}")
