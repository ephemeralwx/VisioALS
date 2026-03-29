export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    // CORS headers (add if your desktop app needs them)
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    try {
      if (path === "/health" && request.method === "GET") {
        return json({ status: "ok" }, corsHeaders);
      }

      if (path === "/telemetry" && request.method === "POST") {
        const { duration_seconds } = await request.json();
        if (typeof duration_seconds !== "number" || duration_seconds <= 0) {
          return json({ error: "invalid duration" }, corsHeaders, 400);
        }
        const stats = JSON.parse((await env.TELEMETRY.get("stats")) || '{"total_sessions":0,"total_duration":0}');
        stats.total_sessions += 1;
        stats.total_duration += duration_seconds;
        await env.TELEMETRY.put("stats", JSON.stringify(stats));
        return json({ ok: true }, corsHeaders);
      }

      if (path === "/stats" && request.method === "GET") {
        const stats = JSON.parse((await env.TELEMETRY.get("stats")) || '{"total_sessions":0,"total_duration":0}');
        const avg = stats.total_sessions > 0 ? stats.total_duration / stats.total_sessions : 0;
        return json({
          total_sessions: stats.total_sessions,
          total_duration_seconds: stats.total_duration,
          avg_session_seconds: Math.round(avg),
          avg_session_minutes: +(avg / 60).toFixed(1),
        }, corsHeaders);
      }

      if (path === "/generate-options" && request.method === "POST") {
        const { question } = await request.json();
        if (!question) return json({ error: "question is required" }, corsHeaders, 400);

        const prompt =
          `A caregiver asked an ALS patient: "${question}"\n\n` +
          "Generate exactly 4 short possible answers the patient might want to give.\n" +
          "Rules:\n" +
          "- Each answer must be a brief phrase (2-8 words).\n" +
          "- All 4 answers MUST be meaningfully different from each other.\n" +
          "- Cover a spread: one positive, one negative, one practical, one emotional.\n" +
          "- No two answers should convey the same sentiment or meaning.\n" +
          "Return ONLY a JSON array of exactly 4 strings. No markdown, no explanation.";

        const raw = await callOpenRouter(env, prompt, 0.9);
        let options = extractJsonArray(raw);
        options = options.map(String);
        while (options.length < 4) options.push("(no response)");
        return json({ options: options.slice(0, 4) }, corsHeaders);
      }

      if (path === "/expand-response" && request.method === "POST") {
        const { question, response } = await request.json();
        if (!question || !response)
          return json({ error: "question and response are required" }, corsHeaders, 400);

        const prompt =
          `A caregiver asked: "${question}"\n` +
          `The ALS patient selected this short answer: "${response}"\n\n` +
          "Turn the patient's short answer into a natural, complete sentence that answers the question. " +
          "Keep it brief (1-2 sentences). Speak from the patient's perspective (first person).";

        const text = await callOpenRouter(env, prompt, 0.7);
        return json({ expanded: text }, corsHeaders);
      }

      return json({ error: "Not found" }, corsHeaders, 404);
    } catch (e) {
      console.error(e);
      return json({ error: e.message }, corsHeaders, 502);
    }
  },
};

async function callOpenRouter(env, userPrompt, temperature) {
  const model = env.OPENROUTER_MODEL || "openai/gpt-4.1-nano";
  const apiKey = env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error("OPENROUTER_API_KEY is not set");

  const systemContent =
    temperature > 0.8
      ? "You respond with raw JSON only. No markdown fences, no explanation."
      : "You are a helpful assistant.";

  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: systemContent },
        { role: "user", content: userPrompt },
      ],
      temperature,
    }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`OpenRouter returned ${res.status}: ${body}`);
  }

  const data = await res.json();
  return data.choices[0].message.content.trim();
}

function extractJsonArray(raw) {
  let cleaned = raw.replace(/^```(?:json)?\s*/g, "").replace(/\s*```$/g, "");
  const parsed = JSON.parse(cleaned);
  if (!Array.isArray(parsed)) throw new Error(`Expected array, got ${typeof parsed}`);
  return parsed;
}

function json(data, corsHeaders, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders },
  });
}