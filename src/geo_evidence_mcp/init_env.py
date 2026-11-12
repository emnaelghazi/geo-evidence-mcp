"""Create .env from .env.example, replacing CHANGEME placeholders with random secrets (R14)."""

from __future__ import annotations

import secrets
from pathlib import Path

PLACEHOLDER = "CHANGEME"
# Keys that need a recognisable prefix.
PREFIXES = {
    "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": "pk-lf-",
    "LANGFUSE_INIT_PROJECT_SECRET_KEY": "sk-lf-",
}
# Client-side keys that must equal the server-side bootstrap keys.
MIRRORS = {
    "LANGFUSE_PUBLIC_KEY": "LANGFUSE_INIT_PROJECT_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY": "LANGFUSE_INIT_PROJECT_SECRET_KEY",
}


def _parse(line: str) -> tuple[str | None, str | None]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None, None
    key, _, value = stripped.partition("=")
    return key.strip(), value.strip()


def render_env(template: str) -> str:
    """Return the template with every CHANGEME value replaced by a generated secret."""
    generated: dict[str, str] = {}
    for line in template.splitlines():
        key, value = _parse(line)
        if key and value == PLACEHOLDER and key not in MIRRORS:
            # 32 random bytes -> 64 hex chars (also satisfies Langfuse's ENCRYPTION_KEY format).
            generated[key] = PREFIXES.get(key, "") + secrets.token_hex(32)

    rendered: list[str] = []
    for line in template.splitlines():
        key, value = _parse(line)
        if key and value == PLACEHOLDER:
            source = MIRRORS.get(key, key)
            line = f"{key}={generated[source]}"
        rendered.append(line)
    return "\n".join(rendered) + "\n"


def main() -> int:
    template_path = Path(".env.example")
    target = Path(".env")
    if target.exists():
        print(".env already exists; not overwriting.")
        return 0
    if not template_path.exists():
        print("Run this from the repository root (.env.example not found).")
        return 1
    target.write_text(render_env(template_path.read_text(encoding="utf-8")), encoding="utf-8")
    target.chmod(0o600)
    print("Wrote .env with generated secrets. Langfuse login: see LANGFUSE_INIT_USER_* in .env.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
