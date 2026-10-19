"""Explicit asyncpg URL conversion without discarding TLS or other options."""
from sqlalchemy.engine import make_url
import ssl


def async_database_options(value: str, *, production: bool = False):
    if not value:
        if production:
            raise ValueError("DATABASE_URL is required outside development")
        return "sqlite+aiosqlite:///./test.db", {}
    url = make_url(value)
    if url.drivername.startswith("sqlite"):
        if production:
            raise ValueError("Production requires a PostgreSQL DATABASE_URL")
        return url, {}
    if url.drivername not in {"postgres", "postgresql", "postgresql+asyncpg"}:
        raise ValueError("Use a PostgreSQL URL with the asyncpg driver")
    query = dict(url.query)
    connect = {"timeout": 10}
    sslmode = query.pop("sslmode", query.pop("ssl", None))
    if sslmode is not None:
        if sslmode not in {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}:
            raise ValueError("Unsupported PostgreSQL sslmode")
        if production and sslmode in {"disable", "allow", "prefer"}:
            raise ValueError("Production PostgreSQL requires TLS")
    elif production:
        sslmode = "verify-full"
    if sslmode in {"require", "verify-ca", "verify-full"}:
        # Use system trust roots by default; asyncpg's string verify-full mode
        # otherwise requires a ~/.postgresql/root.crt even for public CAs.
        context = ssl.create_default_context(cafile=query.pop("sslrootcert", None))
        context.check_hostname = sslmode == "verify-full"
        if sslmode == "require":
            context.verify_mode = ssl.CERT_NONE
        cert = query.pop("sslcert", None)
        key = query.pop("sslkey", None)
        if cert:
            context.load_cert_chain(cert, keyfile=key)
        elif key:
            raise ValueError("sslkey requires sslcert")
        connect["ssl"] = context
    elif sslmode:
        connect["ssl"] = sslmode
    binding = query.pop("channel_binding", None)
    if binding not in (None, "disable", "prefer"):
        raise ValueError("asyncpg does not support channel_binding=require; configure supported TLS verification explicitly")
    if "connect_timeout" in query:
        connect["timeout"] = float(query.pop("connect_timeout"))
    if "application_name" in query:
        connect["server_settings"] = {"application_name": query.pop("application_name")}
    return url.set(drivername="postgresql+asyncpg", query=query), connect
