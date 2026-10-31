/**
 * Centralized API base URL.
 * In production (Vercel), defaults to '' so all requests use relative paths (/api/...)
 * matching the same-origin Serverless Functions without CORS or mixed content issues.
 * In development, Vite's dev proxy forwards /api to http://127.0.0.1:8000 automatically.
 */
export const API_URL = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')
