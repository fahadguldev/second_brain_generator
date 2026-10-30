# Second Brain backend

## Chat persistence

The API assigns anonymous visitors a signed, HTTP-only cookie and stores their
conversation history in the application database. SQLite is used locally by
default; set `DATABASE_URL` to a PostgreSQL/Supabase connection string in
production.

Required production configuration:

```env
DATABASE_URL=postgresql://postgres:password@host:5432/postgres
SESSION_SECRET=replace-with-a-long-random-secret
COOKIE_SECURE=true
COOKIE_SAMESITE=none
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-publishable-or-anon-key
ADMIN_EMAILS=admin@example.com
```

Use `COOKIE_SAMESITE=lax` when the frontend and API share a site. If they are
on different sites, HTTPS plus `COOKIE_SAMESITE=none` and `COOKIE_SECURE=true`
are required for browsers to retain the visitor cookie.

Run locally with:

```shell
uv sync
uv run uvicorn src.main:app --reload
```

The `/api/admin/*` routes require a valid Supabase access token and restrict
access to `ADMIN_EMAILS`. Text/Markdown uploads enter as drafts; an admin must
approve them before starting an ingestion job. Qdrant remains a derived index,
while job and chunk state are stored in PostgreSQL.
