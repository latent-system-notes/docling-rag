from ..config import config, get_logger

logger = get_logger(__name__)


def ldap_authenticate(username: str, password: str) -> bool:
    """Authenticate user against LDAP/AD. Returns True if credentials are valid."""
    ldap_url = config("LDAP_URL")
    if not ldap_url:
        logger.warning("LDAP_URL not configured")
        return False

    try:
        import ldap3
        base_dn = config("LDAP_BASE_DN") or ""
        bind_dn_template = config("LDAP_BIND_DN") or f"cn={{}},{base_dn}"
        bind_dn = bind_dn_template.format(username) if "{}" in bind_dn_template else bind_dn_template

        server = ldap3.Server(ldap_url, get_info=ldap3.NONE, connect_timeout=10)
        conn = ldap3.Connection(server, user=bind_dn, password=password, auto_bind=True, read_only=True,
                                receive_timeout=10)
        conn.unbind()
        logger.info(f"LDAP auth success: {username}")
        return True
    except ImportError:
        logger.error("ldap3 package not installed")
        return False
    except Exception as e:
        logger.info(f"LDAP auth failed for {username}: {e}")
        return False
