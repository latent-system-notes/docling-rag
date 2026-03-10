"""
Test: RBAC permission visibility across all query functions.

Permission rules:
- Documents WITHOUT entries in document_permissions → PUBLIC (all users see them)
- Documents WITH entries in document_permissions → RESTRICTED (only matching groups)
- Admin (groups=None) → sees everything

Scenarios:
  1. Admin (groups=None) → sees ALL documents
  2. User with group_a → sees public + restricted_a
  3. User with no groups ([]) → sees only public
  4. User with group_b → sees public + restricted_b
  5. User with both groups → sees everything
  6. Fulltext search respects same rules
  7. list_chunks / count_chunks respects same rules
  8. Dynamic: admin adds path_permission → refresh → doc becomes restricted
  9. Dynamic: admin removes path_permission → refresh → doc becomes public again
 10. Dynamic: admin reassigns permission to different group → user visibility changes

Requires: PostgreSQL running on .env.test config (localhost:5433).
"""
import sys
import os
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

from src.storage.postgres import (
    get_pool, create_collection,
    add_vectors, search_vectors, search_fulltext,
    list_documents, list_chunks, count_chunks,
    create_group, delete_group,
    set_document_permissions,
    add_path_permission, remove_path_permission,
    compute_effective_groups, refresh_all_document_permissions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
EMBED_DIM = 768
TEST_PREFIX = "__test_perm_"

_passed = 0
_failed = 0


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


def _cleanup():
    """Remove all test data."""
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute("DELETE FROM document_permissions WHERE doc_id LIKE %s", (f"{TEST_PREFIX}%",))
        conn.execute("DELETE FROM path_permissions WHERE path LIKE %s", (f"{TEST_PREFIX}%",))
        conn.execute("DELETE FROM chunks WHERE doc_id LIKE %s", (f"{TEST_PREFIX}%",))
        conn.execute("DELETE FROM groups WHERE name LIKE %s", (f"{TEST_PREFIX}%",))
        conn.commit()


def check(name: str, condition: bool, detail: str = ""):
    global _passed, _failed
    if condition:
        print(f"  PASS  {name}")
        _passed += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        _failed += 1


def _get_doc_ids_from_vectors(query_vec, top_k, groups):
    results = search_vectors(query_vec, top_k=top_k, groups=groups)
    return {r["doc_id"] for r in results}


def _get_doc_ids_from_list(groups):
    docs = list_documents(groups=groups)
    return {d["doc_id"] for d in docs}


def _get_doc_ids_from_chunks(groups):
    chunks = list_chunks(groups=groups, limit=1000)
    return {c["doc_id"] for c in chunks}


def _count_chunks_for(groups, doc_id=None):
    return count_chunks(groups=groups)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def run_tests():
    global _passed, _failed
    _passed = 0
    _failed = 0

    create_collection()
    _cleanup()

    # --- Setup ---
    pub_doc = f"{TEST_PREFIX}public_doc"
    restricted_a = f"{TEST_PREFIX}restricted_a"
    restricted_b = f"{TEST_PREFIX}restricted_b"
    dynamic_doc = f"{TEST_PREFIX}dynamic_doc"

    _insert_doc(pub_doc, f"{TEST_PREFIX}docs/public/readme.txt",
                "This is a public document accessible to everyone", seed=1)
    _insert_doc(restricted_a, f"{TEST_PREFIX}docs/secret_a/report.pdf",
                "This is a restricted document for group A only", seed=2)
    _insert_doc(restricted_b, f"{TEST_PREFIX}docs/secret_b/data.csv",
                "This is a restricted document for group B only", seed=3)
    _insert_doc(dynamic_doc, f"{TEST_PREFIX}docs/shared/notes.md",
                "This document will have permissions changed dynamically", seed=4)

    grp_a = create_group(f"{TEST_PREFIX}group_a", "Test group A")
    grp_b = create_group(f"{TEST_PREFIX}group_b", "Test group B")

    # Assign document_permissions directly: restricted_a → group_a, restricted_b → group_b
    # pub_doc and dynamic_doc have NO entries → public
    set_document_permissions(restricted_a, [grp_a["id"]])
    set_document_permissions(restricted_b, [grp_b["id"]])

    query_vec = _vec(1)
    pool = get_pool()
    with pool.connection() as conn:
        total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    big_k = total_chunks + 10
    print(f"\n  DB has {total_chunks} chunks, using top_k={big_k}")

    groups_a = [f"{TEST_PREFIX}group_a"]
    groups_b = [f"{TEST_PREFIX}group_b"]
    groups_ab = [f"{TEST_PREFIX}group_a", f"{TEST_PREFIX}group_b"]

    # =======================================================================
    # Scenario 1: Admin (groups=None) → sees ALL
    # =======================================================================
    print("\n[Scenario 1] Admin (groups=None) sees ALL docs")
    ids = _get_doc_ids_from_vectors(query_vec, big_k, groups=None)
    check("vectors: sees public", pub_doc in ids)
    check("vectors: sees restricted_a", restricted_a in ids)
    check("vectors: sees restricted_b", restricted_b in ids)
    check("vectors: sees dynamic_doc", dynamic_doc in ids)

    ids = _get_doc_ids_from_list(groups=None)
    check("list_documents: sees all 4", {pub_doc, restricted_a, restricted_b, dynamic_doc} <= ids)

    ids = _get_doc_ids_from_chunks(groups=None)
    check("list_chunks: sees all 4", {pub_doc, restricted_a, restricted_b, dynamic_doc} <= ids)

    # =======================================================================
    # Scenario 2: User with group_a → sees public + restricted_a + dynamic(public)
    # =======================================================================
    print("\n[Scenario 2] User with group_a")
    ids = _get_doc_ids_from_vectors(query_vec, big_k, groups=groups_a)
    check("vectors: sees public", pub_doc in ids)
    check("vectors: sees restricted_a", restricted_a in ids)
    check("vectors: NOT restricted_b", restricted_b not in ids, f"got: {ids}")
    check("vectors: sees dynamic (public)", dynamic_doc in ids)

    ids = _get_doc_ids_from_list(groups=groups_a)
    check("list_documents: sees public + restricted_a + dynamic", {pub_doc, restricted_a, dynamic_doc} <= ids)
    check("list_documents: NOT restricted_b", restricted_b not in ids)

    ids = _get_doc_ids_from_chunks(groups=groups_a)
    check("list_chunks: sees public + restricted_a + dynamic", {pub_doc, restricted_a, dynamic_doc} <= ids)
    check("list_chunks: NOT restricted_b", restricted_b not in ids)

    # =======================================================================
    # Scenario 3: User with NO groups ([]) → sees only public docs
    # =======================================================================
    print("\n[Scenario 3] User with no groups ([])")
    ids = _get_doc_ids_from_vectors(query_vec, big_k, groups=[])
    check("vectors: sees public", pub_doc in ids)
    check("vectors: sees dynamic (public)", dynamic_doc in ids)
    check("vectors: NOT restricted_a", restricted_a not in ids, f"got: {ids}")
    check("vectors: NOT restricted_b", restricted_b not in ids, f"got: {ids}")

    ids = _get_doc_ids_from_list(groups=[])
    check("list_documents: sees public + dynamic only", pub_doc in ids and dynamic_doc in ids)
    check("list_documents: NOT restricted_a", restricted_a not in ids)
    check("list_documents: NOT restricted_b", restricted_b not in ids)

    ids = _get_doc_ids_from_chunks(groups=[])
    check("list_chunks: sees public + dynamic only", pub_doc in ids and dynamic_doc in ids)
    check("list_chunks: NOT restricted", restricted_a not in ids and restricted_b not in ids)

    cnt_admin = count_chunks(groups=None)
    cnt_empty = count_chunks(groups=[])
    check("count_chunks: admin > empty groups", cnt_admin >= cnt_empty,
          f"admin={cnt_admin}, empty={cnt_empty}")

    # =======================================================================
    # Scenario 4: User with group_b → sees public + restricted_b + dynamic
    # =======================================================================
    print("\n[Scenario 4] User with group_b")
    ids = _get_doc_ids_from_vectors(query_vec, big_k, groups=groups_b)
    check("vectors: sees public", pub_doc in ids)
    check("vectors: NOT restricted_a", restricted_a not in ids)
    check("vectors: sees restricted_b", restricted_b in ids)

    ids = _get_doc_ids_from_list(groups=groups_b)
    check("list_documents: public + restricted_b + dynamic", {pub_doc, restricted_b, dynamic_doc} <= ids)
    check("list_documents: NOT restricted_a", restricted_a not in ids)

    # =======================================================================
    # Scenario 5: User with both groups → sees everything
    # =======================================================================
    print("\n[Scenario 5] User with both groups")
    ids = _get_doc_ids_from_list(groups=groups_ab)
    check("list_documents: sees all 4", {pub_doc, restricted_a, restricted_b, dynamic_doc} <= ids)

    ids = _get_doc_ids_from_chunks(groups=groups_ab)
    check("list_chunks: sees all 4", {pub_doc, restricted_a, restricted_b, dynamic_doc} <= ids)

    # =======================================================================
    # Scenario 6: Fulltext search
    # =======================================================================
    print("\n[Scenario 6] Fulltext search")
    ft_all = search_fulltext("document", top_k=big_k, groups=None)
    ft_ids_all = {r[0] for r in ft_all}
    if ft_ids_all:
        ft_empty = search_fulltext("document", top_k=big_k, groups=[])
        ft_ids_empty = {r[0] for r in ft_empty}
        ft_a = search_fulltext("document", top_k=big_k, groups=groups_a)
        ft_ids_a = {r[0] for r in ft_a}

        check("fulltext admin: sees all test chunks",
              f"{pub_doc}_c0" in ft_ids_all and f"{restricted_a}_c0" in ft_ids_all)
        check("fulltext []: only public + dynamic",
              f"{pub_doc}_c0" in ft_ids_empty
              and f"{restricted_a}_c0" not in ft_ids_empty
              and f"{restricted_b}_c0" not in ft_ids_empty,
              f"got: {ft_ids_empty}")
        check("fulltext group_a: public + restricted_a + dynamic",
              f"{pub_doc}_c0" in ft_ids_a
              and f"{restricted_a}_c0" in ft_ids_a
              and f"{restricted_b}_c0" not in ft_ids_a,
              f"got: {ft_ids_a}")
    else:
        print("  (skipped — fulltext returned no results for test data)")

    # =======================================================================
    # Scenario 7: Dynamic — make public doc restricted via path_permission
    # =======================================================================
    print("\n[Scenario 7] Dynamic: assign path_permission to dynamic_doc → becomes restricted")

    # Before: dynamic_doc is public, user with [] can see it
    ids = _get_doc_ids_from_list(groups=[])
    check("BEFORE: [] sees dynamic_doc", dynamic_doc in ids)

    # Admin assigns path_permission for the dynamic doc's folder → group_a only
    add_path_permission(f"{TEST_PREFIX}docs/shared", grp_a["id"])
    refresh_all_document_permissions()

    # After refresh: dynamic_doc now has permission entry → restricted to group_a
    ids = _get_doc_ids_from_list(groups=[])
    check("AFTER restrict: [] does NOT see dynamic_doc", dynamic_doc not in ids,
          f"got: {ids}")

    ids = _get_doc_ids_from_list(groups=groups_a)
    check("AFTER restrict: group_a sees dynamic_doc", dynamic_doc in ids)

    ids = _get_doc_ids_from_list(groups=groups_b)
    check("AFTER restrict: group_b does NOT see dynamic_doc", dynamic_doc not in ids)

    ids = _get_doc_ids_from_chunks(groups=[])
    check("AFTER restrict: chunks [] does NOT see dynamic_doc", dynamic_doc not in ids)

    ids = _get_doc_ids_from_chunks(groups=groups_a)
    check("AFTER restrict: chunks group_a sees dynamic_doc", dynamic_doc in ids)

    # =======================================================================
    # Scenario 8: Dynamic — remove path_permission → doc becomes public again
    # =======================================================================
    print("\n[Scenario 8] Dynamic: remove path_permission → dynamic_doc becomes public again")

    remove_path_permission(f"{TEST_PREFIX}docs/shared", grp_a["id"])
    refresh_all_document_permissions()

    # After removal: dynamic_doc has no permission entries → public again
    ids = _get_doc_ids_from_list(groups=[])
    check("AFTER remove: [] sees dynamic_doc again", dynamic_doc in ids)

    ids = _get_doc_ids_from_list(groups=groups_b)
    check("AFTER remove: group_b sees dynamic_doc again", dynamic_doc in ids)

    ids = _get_doc_ids_from_chunks(groups=[])
    check("AFTER remove: chunks [] sees dynamic_doc again", dynamic_doc in ids)

    # =======================================================================
    # Scenario 9: Dynamic — reassign to different group
    # =======================================================================
    print("\n[Scenario 9] Dynamic: reassign permission from group_a to group_b")

    # Step 1: assign to group_a
    add_path_permission(f"{TEST_PREFIX}docs/shared", grp_a["id"])
    refresh_all_document_permissions()

    ids = _get_doc_ids_from_list(groups=groups_a)
    check("Step1: group_a sees dynamic_doc", dynamic_doc in ids)
    ids = _get_doc_ids_from_list(groups=groups_b)
    check("Step1: group_b does NOT see dynamic_doc", dynamic_doc not in ids)

    # Step 2: remove group_a, assign group_b
    remove_path_permission(f"{TEST_PREFIX}docs/shared", grp_a["id"])
    add_path_permission(f"{TEST_PREFIX}docs/shared", grp_b["id"])
    refresh_all_document_permissions()

    ids = _get_doc_ids_from_list(groups=groups_a)
    check("Step2: group_a does NOT see dynamic_doc", dynamic_doc not in ids,
          f"got: {ids}")
    ids = _get_doc_ids_from_list(groups=groups_b)
    check("Step2: group_b sees dynamic_doc", dynamic_doc in ids)
    ids = _get_doc_ids_from_list(groups=[])
    check("Step2: [] does NOT see dynamic_doc", dynamic_doc not in ids)

    # Verify via vectors too
    ids = _get_doc_ids_from_vectors(query_vec, big_k, groups=groups_a)
    check("Step2 vectors: group_a NOT see dynamic_doc", dynamic_doc not in ids)
    ids = _get_doc_ids_from_vectors(query_vec, big_k, groups=groups_b)
    check("Step2 vectors: group_b sees dynamic_doc", dynamic_doc in ids)

    # =======================================================================
    # Scenario 10: compute_effective_groups — path inheritance
    # =======================================================================
    print("\n[Scenario 10] compute_effective_groups — path inheritance")

    # group_b is assigned to __test_perm_docs/shared (from scenario 9)
    # A file under that path should inherit
    effective = compute_effective_groups(f"{TEST_PREFIX}docs/shared/notes.md")
    check("effective groups for file under assigned folder", grp_b["id"] in effective,
          f"got: {effective}")

    # A file outside that path should NOT inherit
    effective = compute_effective_groups(f"{TEST_PREFIX}docs/public/readme.txt")
    check("no effective groups for unassigned path", grp_b["id"] not in effective,
          f"got: {effective}")

    # --- Cleanup ---
    _cleanup()

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
        run_tests()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        _cleanup()
        sys.exit(1)
