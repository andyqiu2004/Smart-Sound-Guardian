from gmssl import sm2, sm3, func
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import os


class SM2Cipher:
    def __init__(self, private_key, public_key):
        self.sm2_crypt = sm2.CryptSM2(public_key=public_key, private_key=private_key)

    def encrypt(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        return self.sm2_crypt.encrypt(data)

    def decrypt(self, encrypted_data):
        return self.sm2_crypt.decrypt(encrypted_data).decode("utf-8")

    def sign(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        random_hex_str = func.random_hex(self.sm2_crypt.para_len)
        return self.sm2_crypt.sign(data, random_hex_str)

    def verify(self, data, signature):
        if isinstance(data, str):
            data = data.encode("utf-8")
        return self.sm2_crypt.verify(signature, data)


class SM3Hasher:
    @staticmethod
    def hash(data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        return sm3.sm3_hash(func.bytes_to_list(data))

    @staticmethod
    def hmac(key, data):
        if isinstance(key, str):
            key = key.encode("utf-8")
        if isinstance(data, str):
            data = data.encode("utf-8")
        block_size = 64
        if len(key) > block_size:
            key = sm3.sm3_hash(func.bytes_to_list(key))
        key = key.ljust(block_size, b"\0")
        o_key_pad = func.bytes_xor(b"\x5c" * block_size, key)
        i_key_pad = func.bytes_xor(b"\x36" * block_size, key)
        return sm3.sm3_hash(
            func.bytes_to_list(
                o_key_pad + sm3.sm3_hash(func.bytes_to_list(i_key_pad + data)).encode()
            )
        )


class SM4Cipher:
    def __init__(self, key=None):
        self.key = key or get_random_bytes(16)
        self.cipher_ecb = AES.new(self.key, AES.MODE_ECB)

    def encrypt_ecb(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        padded_data = pad(data, AES.block_size)
        return self.cipher_ecb.encrypt(padded_data)

    def decrypt_ecb(self, encrypted_data):
        decrypted_data = self.cipher_ecb.decrypt(encrypted_data)
        return unpad(decrypted_data, AES.block_size).decode("utf-8")

    def encrypt_cbc(self, data, iv=None):
        if isinstance(data, str):
            data = data.encode("utf-8")
        iv = iv or get_random_bytes(AES.block_size)
        cipher_cbc = AES.new(self.key, AES.MODE_CBC, iv)
        padded_data = pad(data, AES.block_size)
        return iv + cipher_cbc.encrypt(padded_data)

    def decrypt_cbc(self, encrypted_data):
        iv = encrypted_data[: AES.block_size]
        cipher_cbc = AES.new(self.key, AES.MODE_CBC, iv)
        decrypted_data = cipher_cbc.decrypt(encrypted_data[AES.block_size :])
        return unpad(decrypted_data, AES.block_size).decode("utf-8")

    def encrypt_ctr(self, data, nonce=None):
        if isinstance(data, str):
            data = data.encode("utf-8")
        nonce = nonce or get_random_bytes(8)
        cipher_ctr = AES.new(self.key, AES.MODE_CTR, nonce=nonce)
        return nonce + cipher_ctr.encrypt(data)

    def decrypt_ctr(self, encrypted_data):
        nonce = encrypted_data[:8]
        cipher_ctr = AES.new(self.key, AES.MODE_CTR, nonce=nonce)
        return cipher_ctr.decrypt(encrypted_data[8:]).decode("utf-8")


class SecurityManager:
    def __init__(self, sm2_private_key, sm2_public_key, sm4_key=None):
        self.sm2_cipher = SM2Cipher(sm2_private_key, sm2_public_key)
        self.sm4_cipher = SM4Cipher(sm4_key)
        self.sm3_hasher = SM3Hasher()

    def encrypt_data(self, data, mode="ECB"):
        if mode == "ECB":
            return self.sm4_cipher.encrypt_ecb(data)
        elif mode == "CBC":
            return self.sm4_cipher.encrypt_cbc(data)
        elif mode == "CTR":
            return self.sm4_cipher.encrypt_ctr(data)
        else:
            raise ValueError("Unsupported encryption mode")

    def decrypt_data(self, encrypted_data, mode="ECB"):
        if mode == "ECB":
            return self.sm4_cipher.decrypt_ecb(encrypted_data)
        elif mode == "CBC":
            return self.sm4_cipher.decrypt_cbc(encrypted_data)
        elif mode == "CTR":
            return self.sm4_cipher.decrypt_ctr(encrypted_data)
        else:
            raise ValueError("Unsupported decryption mode")

    def sign_data(self, data):
        return self.sm2_cipher.sign(data)

    def verify_signature(self, data, signature):
        return self.sm2_cipher.verify(data, signature)

    def hash_data(self, data):
        return self.sm3_hasher.hash(data)

    def hmac_data(self, key, data):
        return self.sm3_hasher.hmac(key, data)
