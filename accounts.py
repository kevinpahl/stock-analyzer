"""User accounts so favorites follow you when you log back in. No Streamlit import.

Passwords are never stored, only a salted PBKDF2 hash. Data lives in Upstash Redis (free tier) when configured.
Otherwise it falls back to a local JSON file: fine on your own PC, but erased when Streamlit Cloud restarts the app.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")
SYMBOL_RE = re.compile(r"^\^?[A-Z0-9]{1,8}(?:[.\-=][A-Z0-9]{1,4})?$")
MIN_PASSWORD, MAX_PASSWORD = 8, 128
PBKDF2_ITERS = 200_000
MAX_FAILS, LOCK_SECONDS = 5, 15 * 60
MAX_FAVS = 50
GENERIC_LOGIN_ERROR = "Wrong username or password."


class AccountError(Exception):
    """A problem worth showing to the user as-is."""


class StoreError(AccountError):
    """The account database couldn't be reached or used."""


def clean_symbols(items) -> list[str]:
    out: list[str] = []
    for s in items or []:
        s = str(s).upper().strip()
        if SYMBOL_RE.match(s) and s not in out:
            out.append(s)
    return out[:MAX_FAVS]


# ---------------------------------------------------------------- storage backends
class UpstashStore:
    """Upstash Redis over its REST API (plain HTTPS, no extra packages)."""
    kind = "Upstash Redis"

    def __init__(self, url: str, token: str):
        self.url, self._token = url.rstrip("/"), token

    def _cmd(self, *args):
        req = urllib.request.Request(
            self.url, data=json.dumps(list(args)).encode(), method="POST",
            headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            raise StoreError(f"The account database refused the request (HTTP {e.code}). "
                             "Check the Upstash URL and token in Secrets.") from None
        except Exception as e:
            raise StoreError(f"Couldn't reach the account database: {type(e).__name__}") from None
        if isinstance(data, dict) and data.get("error"):
            raise StoreError(f"Account database error: {data['error']}")
        return data.get("result") if isinstance(data, dict) else None

    def get(self, key):
        return self._cmd("GET", key)

    def set(self, key, value):
        self._cmd("SET", key, value)

    def set_new(self, key, value) -> bool:
        return self._cmd("SET", key, value, "NX") == "OK"

    def delete(self, key):
        self._cmd("DEL", key)

    def count(self, key) -> int:
        v = self._cmd("GET", key)
        return int(v) if v else 0

    def bump(self, key, seconds) -> int:
        n = int(self._cmd("INCR", key))
        if n == 1:
            self._cmd("EXPIRE", key, seconds)
        return n


class FileStore:
    """JSON file on this computer's disk."""
    kind = "local file"
    _lock = threading.Lock()

    def __init__(self, path):
        self.path = Path(path)

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text("utf-8"))
        except FileNotFoundError:
            return {}
        except Exception:
            raise StoreError("The local account file is unreadable.") from None

    def _save(self, d: dict):
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(d), "utf-8")
            os.replace(tmp, self.path)
        except Exception as e:
            raise StoreError(f"Couldn't write the local account file: {type(e).__name__}") from None

    def _live(self, d: dict, key):
        item = d.get(key)
        if item is None:
            return None
        if item.get("exp") and item["exp"] < time.time():
            d.pop(key, None)
            return None
        return item

    def get(self, key):
        with self._lock:
            item = self._live(self._load(), key)
            return None if item is None else item["v"]

    def set(self, key, value):
        with self._lock:
            d = self._load()
            old = self._live(d, key)
            d[key] = {"v": value, "exp": old.get("exp") if old else None}
            self._save(d)

    def set_new(self, key, value) -> bool:
        with self._lock:
            d = self._load()
            if self._live(d, key) is not None:
                return False
            d[key] = {"v": value, "exp": None}
            self._save(d)
            return True

    def delete(self, key):
        with self._lock:
            d = self._load()
            if d.pop(key, None) is not None:
                self._save(d)

    def count(self, key) -> int:
        v = self.get(key)
        return int(v) if v else 0

    def bump(self, key, seconds) -> int:
        with self._lock:
            d = self._load()
            item = self._live(d, key)
            n = (int(item["v"]) if item else 0) + 1
            d[key] = {"v": str(n), "exp": item["exp"] if item else time.time() + seconds}
            self._save(d)
            return n


# ---------------------------------------------------------------- accounts
class Accounts:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def _norm(username: str) -> str:
        u = (username or "").strip().lower()
        if not USERNAME_RE.match(u):
            raise AccountError("Usernames are 3 to 32 characters: letters, numbers, dots, dashes or underscores.")
        return u

    @staticmethod
    def _hash(password: str, salt: bytes, iters: int = PBKDF2_ITERS) -> str:
        return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iters).hex()

    @staticmethod
    def _check_rules(password: str):
        if not (MIN_PASSWORD <= len(password or "") <= MAX_PASSWORD):
            raise AccountError(f"Passwords must be {MIN_PASSWORD} to {MAX_PASSWORD} characters.")

    @staticmethod
    def _ukey(u): return f"sa:user:{u}"

    @staticmethod
    def _fkey(u): return f"sa:fail:{u}"

    def _record(self, u):
        raw = self.store.get(self._ukey(u))
        return json.loads(raw) if raw else None

    def _verify(self, username: str, password: str):
        """Check a password, counting failures (5 wrong tries locks the account for 15 minutes)."""
        try:
            u = self._norm(username)
        except AccountError:
            raise AccountError(GENERIC_LOGIN_ERROR) from None
        if self.store.count(self._fkey(u)) >= MAX_FAILS:
            raise AccountError("Too many failed attempts for this account. Try again in 15 minutes.")
        rec = self._record(u)
        salt = bytes.fromhex(rec["salt"]) if rec else b"\0" * 16   # same work either way, so timing doesn't leak
        h = self._hash(password or "", salt, rec["iters"] if rec else PBKDF2_ITERS)
        if rec is None or not hmac.compare_digest(h, rec["hash"]):
            self.store.bump(self._fkey(u), LOCK_SECONDS)
            raise AccountError(GENERIC_LOGIN_ERROR)
        self.store.delete(self._fkey(u))
        return u, rec

    def create(self, username: str, password: str, favorites=()) -> tuple[str, list[str]]:
        u = self._norm(username)
        self._check_rules(password)
        salt = secrets.token_bytes(16)
        rec = {"salt": salt.hex(), "hash": self._hash(password, salt), "iters": PBKDF2_ITERS,
               "favorites": clean_symbols(favorites), "created": int(time.time())}
        if not self.store.set_new(self._ukey(u), json.dumps(rec)):
            raise AccountError("That username is taken. Try another.")
        return u, rec["favorites"]

    def login(self, username: str, password: str) -> tuple[str, list[str]]:
        u, rec = self._verify(username, password)
        return u, clean_symbols(rec.get("favorites"))

    def set_favorites(self, username: str, favorites) -> None:
        u = self._norm(username)
        rec = self._record(u)
        if rec is None:
            raise AccountError("This account no longer exists.")
        rec["favorites"] = clean_symbols(favorites)
        self.store.set(self._ukey(u), json.dumps(rec))

    def change_password(self, username: str, old: str, new: str) -> None:
        u, rec = self._verify(username, old)
        self._check_rules(new)
        salt = secrets.token_bytes(16)
        rec.update(salt=salt.hex(), hash=self._hash(new, salt), iters=PBKDF2_ITERS)
        self.store.set(self._ukey(u), json.dumps(rec))

    def delete(self, username: str, password: str) -> None:
        u, _ = self._verify(username, password)
        self.store.delete(self._ukey(u))


def build_accounts(url: str | None, token: str | None, local_path) -> Accounts:
    if url and token:
        return Accounts(UpstashStore(url, token))
    return Accounts(FileStore(local_path))
