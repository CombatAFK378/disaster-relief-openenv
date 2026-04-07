# server/app.py - OpenEnv required entry point
import os

import uvicorn

from app.main import app


def main() -> None:
	port = int(os.getenv("PORT", "8000"))
	uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
	main()


__all__ = ["app", "main"]
