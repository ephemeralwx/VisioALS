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
        const {
          question, history = [], rejected = [],
          linguistic_profile_summary = null,
          exemplars = null,
          preference_rules = null,
        } = await request.json();
        if (!question) return json({ error: "question is required" }, corsHeaders, 400);

        let historyBlock = "";
        if (history.length > 0) {
          const pairs = history.slice(-5);
          const lines = pairs.map(p => `  Q: "${p.question}" → A: "${p.answer}"`);
          historyBlock = "Recent conversation:\n" + lines.join("\n") + "\n\n";
        }

        let rejectedBlock = "";
        if (rejected.length > 0) {
          rejectedBlock =
            "The patient already rejected these options — do NOT repeat or rephrase them:\n" +
            rejected.map(r => `"${r}"`).join(", ") + "\n" +
            "Generate completely different answers.\n\n";
        }

        const identityBlock = buildIdentityBlock(linguistic_profile_summary, exemplars, preference_rules);

        const prompt =
          historyBlock +
          identityBlock +
          `A caregiver asked an ALS patient: "${question}"\n\n` +
          rejectedBlock +
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
        const {
          question, response, history = [],
          linguistic_profile_summary = null,
          exemplars = null,
        } = await request.json();
        if (!question || !response)
          return json({ error: "question and response are required" }, corsHeaders, 400);

        let historyBlock = "";
        if (history.length > 0) {
          const pairs = history.slice(-5);
          const lines = pairs.map(p => `  Q: "${p.question}" → A: "${p.answer}"`);
          historyBlock = "Recent conversation:\n" + lines.join("\n") + "\n\n";
        }

        const identityBlock = buildIdentityBlock(linguistic_profile_summary, exemplars);

        const prompt =
          historyBlock +
          identityBlock +
          `A caregiver asked: "${question}"\n` +
          `The ALS patient selected this short answer: "${response}"\n\n` +
          "Turn the patient's short answer into a natural, complete sentence that answers the question. " +
          "Keep it brief (1-2 sentences). Speak from the patient's perspective (first person). " +
          "Match the patient's voice if a communication model is provided above.";

        const text = await callOpenRouter(env, prompt, 0.7);
        return json({ expanded: text }, corsHeaders);
      }

      if (path === "/analyze-style" && request.method === "POST") {
        const { sample_texts = [] } = await request.json();
        const textsBlock = sample_texts.slice(0, 50).join("\n---\n");
        const prompt =
          "Analyze the following writing samples from a single person. " +
          "Characterize their communication style.\n\n" +
          `Samples:\n${textsBlock}\n\n` +
          "Respond with a JSON object containing exactly these fields:\n" +
          '- "humor_style": describe their humor style (e.g. "dry", "sarcastic", "self-deprecating", "none")\n' +
          '- "tone_description": describe their overall tone in 3-6 words\n' +
          '- "emotional_valence": do they tend toward "positive", "negative", or "neutral" framing?\n' +
          '- "personality_notes": 1-2 sentences about notable personality traits in their writing\n' +
          "Return ONLY the JSON object. No markdown, no explanation.";

        const raw = await callOpenRouter(env, prompt, 0.3);
        const cleaned = raw.replace(/^```(?:json)?\s*/g, "").replace(/\s*```$/g, "");
        const data = JSON.parse(cleaned);
        return json({
          humor_style: data.humor_style || "unknown",
          tone_description: data.tone_description || "unknown",
          emotional_valence: data.emotional_valence || "neutral",
          personality_notes: data.personality_notes || "",
        }, corsHeaders);
      }

      if (path === "/analyze-preferences" && request.method === "POST") {
        const { interactions = [] } = await request.json();
        const logLines = interactions.slice(-100).map(e => {
          const selected = e.selected || "(none — all rejected)";
          const rejected = (e.rejected || []).map(r => `"${r}"`).join(", ");
          return `Q: "${e.question || ""}"\n  Selected: "${selected}"\n  Rejected: [${rejected}]`;
        });
        const prompt =
          "Here is a log of an ALS patient's response selections and rejections " +
          "in a communication aid. The most recent interactions are at the end — " +
          "prioritize recent patterns over older ones.\n\n" +
          logLines.join("\n\n") + "\n\n" +
          "Analyze the patterns in what they reject vs. what they select. " +
          "Extract 3-8 concrete preference rules. Each rule should describe a " +
          "pattern, not a specific response.\n\n" +
          "Examples of good rules:\n" +
          '- "Avoids sentimental or emotionally effusive language"\n' +
          '- "Prefers responses under 5 words"\n' +
          '- "Tends to pick the most direct/practical option"\n\n' +
          "Return ONLY a JSON array of rule strings. No markdown, no explanation.";

        const raw = await callOpenRouter(env, prompt, 0.3);
        const rules = extractJsonArray(raw).map(String).slice(0, 8);
        return json({ rules }, corsHeaders);
      }

      return json({ error: "Not found" }, corsHeaders, 404);
    } catch (e) {
      console.error(e);
      return json({ error: e.message }, corsHeaders, 502);
    }
  },
};

function buildIdentityBlock(profileSummary, exemplars, preferenceRules) {
  if (!profileSummary && !exemplars && !preferenceRules) return "";
  const parts = [];
  if (profileSummary) parts.push(`Voice: ${profileSummary}`);
  if (preferenceRules && preferenceRules.length > 0) {
    parts.push("Preferences:\n" + preferenceRules.map(r => `- ${r}`).join("\n"));
  }
  if (exemplars && exemplars.length > 0) {
    parts.push(
      "Style examples from the patient's past writing:\n" +
      exemplars.slice(0, 5).map(e => `  "${e}"`).join("\n")
    );
  }
  return "Patient Communication Model:\n" + parts.join("\n") +
    "\nGenerated answers MUST match this patient's voice and preferences.\n\n";
}

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