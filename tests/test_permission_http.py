"""
Test: RBAC permission visibility via HTTP API (full stack).

Tests the ENTIRE flow through FastAPI routes, JWT auth, and storage.
Uses httpx AsyncClient with ASGI transport — no running server needed.

Scenarios:
  1. Admin sees all docs/chunks
  2. User with group → sees public + matching restricted
  3. User with no group → sees only public
  4. Dynamic: admin adds path_permission → user visibility changes (re-login for fresh token)
  5. Dynamic: admin removes path_permission → doc becomes public again
  6. Dynamic: admin reassigns group → user sees different docs
  7. Admin changes user's groups → re-login reflects new groups

Requires: PostgreSQL running on .env.test config.
"""
import sys
import asyncio
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env.test", override=True)
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from src.config import _setup_hf_env
_setup_hf_env()

import httpx
from src.dashboard.app import create_app
from src.auth.auth import hash_password
from src.storage.postgres import (
    get_pool, create_collection, add_vectors,
    add_path_permission, refresh_all_document_permissions,
    create_user, create_group as db_create_group,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBED_DIM = 768
TEST_PREFIX = "__test_http_"
ADMIN_USER = f"{TEST_PREFIX}admin"
ADMIN_PASS = "adminpass123"
TEST_USER = f"{TEST_PREFIX}user"
TEST_PASS = "testpass123"

_passed = 0
_failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global _passed, _failed
    if condition:
        print(f"  PASS  {name}")
        _passed += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        _failed += 1


def _vec(seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(EMBED_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _insert_doc(doc_id: str, file_path: str, text: str, seed: int = 0):
    chunk_id = f"{doc_id}_c0"
    vec = _vec(seed).reshape(1, -1)
    meta = {
        "doc_id": doc_id,
        "text": text,
        "page_num": 1,
        "doc_type": "test",
        "language": "en",
        "file_path": file_path,
        "ingested_at": "2026-01-01T00:00:00",
        "chunk_index": 0,
    }
    add_vectors([chunk_id], vec, [meta])


def _db_cleanup():
    """Remove all test data from DB."""
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute("DELETE FROM document_permissions WHERE doc_id LIKE %s", (f"{TEST_PREFIX}%",))
        conn.execute("DELETE FROM path_permissions WHERE path LIKE %s", (f"{TEST_PREFIX}%",))
        conn.execute("DELETE FROM chunks WHERE doc_id LIKE %s", (f"{TEST_PREFIX}%",))
        # Remove test user's group assignments before deleting user/groups
        conn.execute(
            "DELETE FROM user_groups WHERE user_id IN (SELECT id FROM users WHERE username LIKE %s)",
            (f"{TEST_PREFIX}%",),
        )
        conn.execute("DELETE FROM users WHERE username LIKE %s", (f"{TEST_PREFIX}%",))
        conn.execute("DELETE FROM groups WHERE name LIKE %s", (f"{TEST_PREFIX}%",))
        conn.commit()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
async def login(client: httpx.AsyncClient, username: str, password: str) -> str:
    """Login and return Bearer token."""
    r = await client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, f"Login failed for {username}: {r.status_code} {r.text}"
    return r.json()["access_token"]


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def get_doc_ids(client: httpx.AsyncClient, token: str, endpoint: str) -> set:
    """Fetch doc_ids from an endpoint."""
    r = await client.get(endpoint, headers=auth(token))
    assert r.status_code == 200, f"{endpoint} failed: {r.status_code} {r.text}"
    data = r.json()
    if "documents" in data:
        return {d["doc_id"] for d in data["documents"]}
    elif "items" in data:
        return {item["doc_id"] for item in data["items"]}
    return set()


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
async def run_tests():
    global _passed, _failed
    _passed = 0
    _failed = 0

    create_collection()
    _db_cleanup()

    # --- Create test users and groups directly via DB ---
    pw_hash = hash_password(ADMIN_PASS)
    admin_user = create_user(ADMIN_USER, pw_hash, "Test Admin", "", is_admin=True)
    print(f"\n  Created admin: id={admin_user['id']}")

    pw_hash = hash_password(TEST_PASS)
    test_user = create_user(TEST_USER, pw_hash, "Test User", "", is_admin=False)
    test_user_id = test_user["id"]
    print(f"  Created test user: id={test_user_id}")

    grp_a = db_create_group(f"{TEST_PREFIX}grp_a", "Test A")
    grp_b = db_create_group(f"{TEST_PREFIX}grp_b", "Test B")
    grp_a_id = grp_a["id"]
    grp_b_id = grp_b["id"]
    print(f"  Created groups: grp_a={grp_a_id}, grp_b={grp_b_id}")

    app = create_app()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as c:

        # --- Login as admin ---
        admin_token = await login(c, ADMIN_USER, ADMIN_PASS)
        print(f"  Admin logged in")

        # --- Insert test documents directly into DB ---
        pub_doc = f"{TEST_PREFIX}public_doc"
        restricted_a = f"{TEST_PREFIX}restricted_a"
        restricted_b = f"{TEST_PREFIX}restricted_b"
        dynamic_doc = f"{TEST_PREFIX}dynamic_doc"

        _insert_doc(pub_doc, f"{TEST_PREFIX}docs/public/readme.txt",
                     "This is a public document for everyone", seed=1)
        _insert_doc(restricted_a, f"{TEST_PREFIX}docs/secret_a/report.pdf",
                     "Restricted document for group A", seed=2)
        _insert_doc(restricted_b, f"{TEST_PREFIX}docs/secret_b/data.csv",
                     "Restricted document for group B", seed=3)
        _insert_doc(dynamic_doc, f"{TEST_PREFIX}docs/shared/notes.md",
                     "Dynamic document that will change permissions", seed=4)

        # Set restrictions via path_permissions (the real workflow)
        # restricted_a folder → grp_a, restricted_b folder → grp_b
        # pub_doc and dynamic_doc → no path_permissions (public)
        add_path_permission(f"{TEST_PREFIX}docs/secret_a", grp_a_id)
        add_path_permission(f"{TEST_PREFIX}docs/secret_b", grp_b_id)
        refresh_all_document_permissions()
        print(f"  Inserted 4 test docs (2 public, 2 restricted via path_permissions)")

        # ===================================================================
        # Scenario 1: Admin sees ALL via /api/search/documents and /api/chunks
        # ===================================================================
        print("\n[Scenario 1] Admin sees ALL")
        ids = await get_doc_ids(c, admin_token, "/api/search/documents?limit=500")
        check("admin /documents: sees public", pub_doc in ids)
        check("admin /documents: sees restricted_a", restricted_a in ids)
        check("admin /documents: sees restricted_b", restricted_b in ids)
        check("admin /documents: sees dynamic", dynamic_doc in ids)

        ids = await get_doc_ids(c, admin_token, "/api/chunks?page_size=100")
        check("admin /chunks: sees public", pub_doc in ids)
        check("admin /chunks: sees restricted_a", restricted_a in ids)
        check("admin /chunks: sees restricted_b", restricted_b in ids)
        check("admin /chunks: sees dynamic", dynamic_doc in ids)

        # ===================================================================
        # Scenario 2: User with NO groups → sees only public
        # ===================================================================
        print("\n[Scenario 2] User with NO groups → only public")
        user_token = await login(c, TEST_USER, TEST_PASS)

        ids = await get_doc_ids(c, user_token, "/api/search/documents?limit=500")
        check("/documents: sees public", pub_doc in ids)
        check("/documents: sees dynamic (public)", dynamic_doc in ids)
        check("/documents: NOT restricted_a", restricted_a not in ids, f"got: {ids}")
        check("/documents: NOT restricted_b", restricted_b not in ids, f"got: {ids}")

        ids = await get_doc_ids(c, user_token, "/api/chunks?page_size=100")
        check("/chunks: sees public", pub_doc in ids)
        check("/chunks: sees dynamic (public)", dynamic_doc in ids)
        check("/chunks: NOT restricted_a", restricted_a not in ids)
        check("/chunks: NOT restricted_b", restricted_b not in ids)

        # ===================================================================
        # Scenario 3: Admin assigns user to grp_a → re-login → sees restricted_a
        # ===================================================================
        print("\n[Scenario 3] Admin assigns grp_a to user → user re-logins")
        r = await c.post(f"/api/users/{test_user_id}/groups",
                         json={"group_id": grp_a_id}, headers=auth(admin_token))
        check("assign grp_a to user", r.status_code == 201, f"status={r.status_code}")

        # Re-login to get token with updated groups
        user_token = await login(c, TEST_USER, TEST_PASS)

        # Verify token has the group
        r = await c.get("/api/auth/me", headers=auth(user_token))
        user_groups = r.json().get("groups", [])
        check("token contains grp_a", f"{TEST_PREFIX}grp_a" in user_groups, f"groups={user_groups}")

        ids = await get_doc_ids(c, user_token, "/api/search/documents?limit=500")
        check("/documents: sees public", pub_doc in ids)
        check("/documents: sees restricted_a", restricted_a in ids)
        check("/documents: NOT restricted_b", restricted_b not in ids, f"got: {ids}")
        check("/documents: sees dynamic (public)", dynamic_doc in ids)

        ids = await get_doc_ids(c, user_token, "/api/chunks?page_size=100")
        check("/chunks: sees public + restricted_a + dynamic",
              pub_doc in ids and restricted_a in ids and dynamic_doc in ids)
        check("/chunks: NOT restricted_b", restricted_b not in ids)

        # ===================================================================
        # Scenario 4: Admin adds path_permission → dynamic_doc becomes restricted
        # ===================================================================
        print("\n[Scenario 4] Admin assigns path_permission → dynamic_doc restricted to grp_b")

        # Before: user (grp_a) sees dynamic_doc (public)
        ids = await get_doc_ids(c, user_token, "/api/search/documents?limit=500")
        check("BEFORE: user sees dynamic_doc", dynamic_doc in ids)

        # Admin adds path_permission for dynamic_doc's folder → grp_b
        # This also triggers refresh_all_document_permissions
        r = await c.post("/api/permissions/paths",
                         json={"path": f"{TEST_PREFIX}docs/shared", "group_id": grp_b_id},
                         headers=auth(admin_token))
        check("add path_permission", r.status_code == 201, f"status={r.status_code} {r.text}")

        # After: user (grp_a) should NOT see dynamic_doc anymore
        ids = await get_doc_ids(c, user_token, "/api/search/documents?limit=500")
        check("AFTER: user does NOT see dynamic_doc", dynamic_doc not in ids, f"got: {ids}")
        check("AFTER: user still sees restricted_a", restricted_a in ids)
        check("AFTER: user still sees public", pub_doc in ids)

        ids = await get_doc_ids(c, user_token, "/api/chunks?page_size=100")
        check("AFTER chunks: dynamic_doc hidden", dynamic_doc not in ids)

        # ===================================================================
        # Scenario 5: Admin removes path_permission → dynamic_doc public again
        # ===================================================================
        print("\n[Scenario 5] Admin removes path_permission → dynamic_doc public again")

        r = await c.request("DELETE", "/api/permissions/paths",
                            json={"path": f"{TEST_PREFIX}docs/shared", "group_id": grp_b_id},
                            headers=auth(admin_token))
        check("remove path_permission", r.status_code == 200, f"status={r.status_code} {r.text}")

        ids = await get_doc_ids(c, user_token, "/api/search/documents?limit=500")
        check("dynamic_doc visible again", dynamic_doc in ids)

        ids = await get_doc_ids(c, user_token, "/api/chunks?page_size=100")
        check("chunks: dynamic_doc visible again", dynamic_doc in ids)

        # ===================================================================
        # Scenario 6: Admin reassigns: dynamic → grp_a, then to grp_b
        # ===================================================================
        print("\n[Scenario 6] Dynamic reassign: grp_a → grp_b")

        # Assign to grp_a → user (grp_a) should see it
        r = await c.post("/api/permissions/paths",
                         json={"path": f"{TEST_PREFIX}docs/shared", "group_id": grp_a_id},
                         headers=auth(admin_token))
        check("assign to grp_a", r.status_code == 201)

        ids = await get_doc_ids(c, user_token, "/api/search/documents?limit=500")
        check("user (grp_a) sees dynamic_doc", dynamic_doc in ids)

        # Remove grp_a, assign grp_b → user (grp_a) should NOT see it
        await c.request("DELETE", "/api/permissions/paths",
                        json={"path": f"{TEST_PREFIX}docs/shared", "group_id": grp_a_id},
                        headers=auth(admin_token))
        r = await c.post("/api/permissions/paths",
                         json={"path": f"{TEST_PREFIX}docs/shared", "group_id": grp_b_id},
                         headers=auth(admin_token))
        check("reassign to grp_b", r.status_code == 201)

        ids = await get_doc_ids(c, user_token, "/api/search/documents?limit=500")
        check("user (grp_a) does NOT see dynamic_doc", dynamic_doc not in ids, f"got: {ids}")

        # ===================================================================
        # Scenario 7: Admin changes user's groups: remove grp_a, add grp_b
        # ===================================================================
        print("\n[Scenario 7] Admin changes user groups: grp_a → grp_b")

        # Remove grp_a from user
        r = await c.delete(f"/api/users/{test_user_id}/groups/{grp_a_id}",
                           headers=auth(admin_token))
        check("remove grp_a from user", r.status_code == 204)

        # Add grp_b to user
        r = await c.post(f"/api/users/{test_user_id}/groups",
                         json={"group_id": grp_b_id}, headers=auth(admin_token))
        check("add grp_b to user", r.status_code == 201)

        # Re-login to get new token with grp_b
        user_token = await login(c, TEST_USER, TEST_PASS)

        r = await c.get("/api/auth/me", headers=auth(user_token))
        user_groups = r.json().get("groups", [])
        check("token now has grp_b", f"{TEST_PREFIX}grp_b" in user_groups, f"groups={user_groups}")
        check("token no longer has grp_a", f"{TEST_PREFIX}grp_a" not in user_groups, f"groups={user_groups}")

        # Now user (grp_b) should see: public, restricted_b, dynamic_doc (assigned to grp_b)
        ids = await get_doc_ids(c, user_token, "/api/search/documents?limit=500")
        check("/documents: sees public", pub_doc in ids)
        check("/documents: sees restricted_b", restricted_b in ids)
        check("/documents: sees dynamic (grp_b)", dynamic_doc in ids)
        check("/documents: NOT restricted_a", restricted_a not in ids, f"got: {ids}")

        ids = await get_doc_ids(c, user_token, "/api/chunks?page_size=100")
        check("/chunks: sees public + restricted_b + dynamic",
              pub_doc in ids and restricted_b in ids and dynamic_doc in ids)
        check("/chunks: NOT restricted_a", restricted_a not in ids)

    # --- Cleanup ---
    _db_cleanup()

    # --- Summary ---
    total = _passed + _failed
    print(f"\n{'=' * 60}")
    print(f"Results: {_passed}/{total} passed, {_failed} failed")
    if _failed:
        print("SOME TESTS FAILED!")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED!")


if __name__ == "__main__":
    try:
        asyncio.run(run_tests())
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        _db_cleanup()
        sys.exit(1)
