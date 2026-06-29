export const UPSTOX_CONFIG = {
  // Set via env (e.g. wrangler secret / .dev.vars); never hardcode tokens.
  ACCESS_TOKEN: process.env.UPSTOX_ACCESS_TOKEN || ""
};