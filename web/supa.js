// One lazily created Supabase client (publishable key, read-only through RLS). Resolves to null when
// the config endpoint, the CDN or the project is unreachable, so every caller has a fallback path.

import { getJSON } from './dom.js';

const SUPABASE_ESM = 'https://esm.sh/@supabase/supabase-js@2.117.1';
let pending = null;

export function getSupabase() {
  if (!pending) {
    pending = (async () => {
      try {
        const cfg = await getJSON('/api/config');
        if (!cfg?.supabase_url || !cfg?.supabase_publishable_key || !/^https:\/\//.test(cfg.supabase_url)) return null;
        const { createClient } = await import(SUPABASE_ESM);
        return createClient(cfg.supabase_url, cfg.supabase_publishable_key, { auth: { persistSession: false, autoRefreshToken: false } });
      } catch {
        return null;
      }
    })();
  }
  return pending;
}
