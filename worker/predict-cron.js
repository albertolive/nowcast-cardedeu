/**
 * Cloudflare Worker that fires `workflow_dispatch` on the GitHub Actions
 * `nowcast.yml` workflow every 10 min. Replaces the ClawCloud container's
 * prediction loop. CF Workers cron triggers are precise (no GH-native cron
 * drift), the free tier covers the load by ~3 orders of magnitude, and the
 * Worker has zero state of its own (the GH workflow holds all state).
 *
 * Configure via wrangler.toml:
 *   - vars.GH_REPO          e.g. "albertolive/nowcast-cardedeu"
 *   - vars.GH_WORKFLOW      e.g. "nowcast.yml"
 *   - vars.GH_REF           e.g. "main"
 *   - secrets.GH_TOKEN      fine-grained PAT, scope = Actions: Read & write
 *                           on this single repo. Set via:
 *                             wrangler secret put GH_TOKEN
 *
 * Failure surface: if the dispatch POST fails, the Worker throws and the
 * error shows in Cloudflare's dashboard. The GH watchdog workflow
 * (.github/workflows/watchdog.yml) is the independent backstop that
 * catches silent failure of *this* Worker by checking prediction freshness
 * once an hour.
 */

const GITHUB_API = "https://api.github.com";

async function dispatchWorkflow(env) {
  const url = `${GITHUB_API}/repos/${env.GH_REPO}/actions/workflows/${env.GH_WORKFLOW}/dispatches`;
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GH_TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "nowcast-cardedeu-cron/1.0",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      ref: env.GH_REF || "main",
      inputs: { action: "predict" },
    }),
  });

  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`workflow_dispatch ${resp.status}: ${body.slice(0, 500)}`);
  }
  return { ok: true, status: resp.status };
}

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(
      dispatchWorkflow(env).then(
        (r) => console.log(`dispatched predict @ ${event.scheduledTime}: ${r.status}`),
        (e) => {
          console.error(e.message);
          throw e;
        },
      ),
    );
  },

  // Manual trigger for smoke-testing: `curl -X POST https://<worker>/dispatch`
  async fetch(req, env) {
    const url = new URL(req.url);
    if (req.method === "POST" && url.pathname === "/dispatch") {
      try {
        const r = await dispatchWorkflow(env);
        return new Response(JSON.stringify(r), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      } catch (e) {
        return new Response(e.message, { status: 502 });
      }
    }
    return new Response("nowcast-cardedeu-cron — POST /dispatch to trigger", {
      status: 200,
    });
  },
};
