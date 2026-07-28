import getpass
import hashlib
import hmac
import os
import sys


def hash_password(password: str, iterations: int = 600_000) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def main() -> int:
    password = getpass.getpass("Password: ")
    confirm = getpass.getpass("Confirm: ")
    if not hmac.compare_digest(password, confirm):
        print("Passwords do not match", file=sys.stderr)
        return 1
    if not password:
        print("Password must not be empty", file=sys.stderr)
        return 1
    print(hash_password(password))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
