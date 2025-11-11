# Ark.CV Auth0 Prototype

This folder hosts a fresh React + Vite app wired to the Auth0 settings you configured for Ark.CV (SPA + RS256 API + RBAC). Use it to verify sign-in, route protection, role claims, and access-token flows before pointing it at the live HITL dashboard.

## Prerequisites

- Node 18+ (25.x installed via Homebrew above)
- Auth0 tenant with:
  - Application (Single Page App) named “ArkCV Web”
  - API identifier `https://api.arkcv.com`
  - Post-login action that writes roles (and optionally `dam`) to `https://arkcv.com/claims/*`

## Setup

1. Copy the sample env file and fill in your tenant values:

   ```bash
   cd ui-console
   cp .env.example .env
   # edit .env with Domain, Client ID, and optional custom scope/namespace values
   ```

   `VITE_AUTH0_APP_BASE_URL` must match an **Allowed Web Origin** and the login callback route (Auth0 is configured for `/callback` per the dashboard instructions).

2. Install deps and start the dev server:

   ```bash
   npm install
   npm run dev
   ```

   The app will be available on the Vite dev URL (default `http://localhost:5173`); ensure this origin is present in Auth0 → Application → Settings (Callback URLs, Logout URLs, Web Origins, CORS).

3. Build for production (optional sanity check):

   ```bash
   npm run build
   npm run preview
   ```

## Key Files

- `src/auth/AuthProvider.jsx` – wraps the SPA with `Auth0Provider`, pulls domain/client ID/audience/scope from env, and enforces the `/callback` redirect required in Auth0’s settings.
- `src/auth/RequireAuth.jsx` – gate for any protected route.
- `src/auth/useRoles.js` – helper that reads `https://arkcv.com/claims/roles` (or any namespace you set via env).
- `src/components/AccessTokenDebugger.jsx` – one-click way to request an access token with the API audience/scopes you defined (`read:clips`, `read:fishcounts`, etc.), mirroring the dashboard’s RBAC setup.
- `src/pages/Callback.jsx` – lightweight landing route for `Allowed Callback URL` completion before navigation resumes.
- `src/pages/Dam.jsx` – demonstrates checking the `…/dam` claim so operators only see their assigned dam’s queue.

## Customizing For Prod

- Swap `VITE_AUTH0_APP_BASE_URL` to your real origin (e.g., `https://app.arkcv.com`) once DNS/certs are ready.
- If you enable Auth0 Organizations per customer, add `organization` to the `authorizationParams` inside `AuthProviderWithNavigate`.
- Update the namespace env (`VITE_AUTH0_NAMESPACE`) if you publish claims on a different URL (e.g., custom domain `https://auth.arkcv.com/claims`).
- Replace the demo dam table with live API calls; the included token debugger already shows how to grab a bearer token.

After you confirm the flows locally, you can fold these Auth0 helpers into the production UI or deploy this Vite build directly behind your CDN.
