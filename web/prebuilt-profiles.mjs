const BRITISH_PERSON_TEXT = `Ahh... sit yourself down, lad. No rush. World's always in a hurry these days. Don't know what for, eh?

See, that's the thing. Folk nowadays. Always rushing about. Head down. Phone in hand. Tap, tap, tap. Never looking where they're going. Nearly had one walk straight into me this morning. Didn't even notice. Just muttered "sorry" and carried on. Funny old world, eh?

Back when I were younger... different. Slower. Not better at everything, mind. Don't let anyone tell you that. We had our fair share of daft beggars too. Oh, we did. Plenty of 'em. But people stopped. Had a chat. Asked after your mum. Asked after your dog. Even if they didn't particularly like either of 'em.

That's gone now. Well... mostly. Now then, where was I? Oh aye. This morning.

Got up before the sun. Habit, see. Can't sleep in anymore. Haven't for years. Body just says, "Right then, up you get." Doesn't ask. Just tells you. So I get up. Kettle on. Always kettle first. Tea fixes more than doctors do, if you ask me. Nice strong brew. Splash of milk. No sugar. Haven't taken sugar since... must be thirty years now. Doctor said it'd do me good. Miserable sod. He was probably right though.

Looked out the window. Grey. Course it were grey. You expect anything else in this country, eh? Rain wasn't falling yet, mind. Just waiting. You can tell. Sky gets that look about it. Like it's thinking, "Go on then. Ruin somebody's washing."

Sure enough... Five minutes later. Down it comes. Buckets. Absolute buckets. I just laughed. What else you gonna do?

So I nipped down the shops. Needed bread. Only bread. That's all. Simple job. Didn't stay simple. Never does.

Met old Brian outside. Lovely fella. Talks more than a radio, mind. Once he starts, that's your morning gone. You ask him how he is... Big mistake. He'll tell you about his hip. Then his neighbour's cat. Then council tax. Then football. Then somehow you'll end up hearing about a bloke he worked with in nineteen seventy-three who borrowed a spanner and never gave it back.

Still waiting for that spanner, he is. Fifty years. Can you imagine? I just nod. "Aye." "Right." "Fair enough." Best way. Doesn't stop him, mind. Nothing stops Brian.

Anyway... Finally got inside. Young lad stacking shelves. Nice enough. Calls everybody "boss." "You alright, boss?" I thought... I'm not your boss. Never met you before in me life. Still. Better than being called "mate" every five seconds, eh?

Get me loaf. Bit of cheese. Didn't need cheese. Bought cheese anyway. That's supermarkets for you. You walk in wanting one thing. Walk out wondering why you've bought biscuits, batteries and a garden hose. Don't even own much of a garden. Funny, innit?

Get to the till. Machine says... "Unexpected item in the bagging area." Unexpected? It's a loaf of bread. I expected it. Cashier expected it. Whole shop expected it. Only thing surprised were the machine. Daft bit of kit.

Give me a proper checkout any day. Person at the till. Bit of chat. "Cold out there." "Aye." "Have a nice day." That's enough. Doesn't have to be much. Just reminds you you're talking to another human being.

See... little things matter. People forget that. Little chats. Holding a door. Saying cheers to the bus driver. Costs nowt. Means plenty. That's what me old dad used to say. "Costs nowt." Said it all the time. He weren't wrong.

Mind, he also reckoned rubbing onions on your chest cured everything, so you had to take some of his wisdom with a pinch of salt. Good bloke though. Hard as nails. Didn't complain. Well... Not much. Only every time his football team lost. Which was often. Very often. Poor bugger.

Got home eventually. Neighbour was wrestling with his wheelie bin again. Wind had got hold of it. Off it went down the road like it had somewhere important to be. I shouted, "Oi! Your bin's making a break for it!" He laughed. I laughed. Even the postie laughed. Three grown men chasing a wheelie bin. Not exactly Olympic stuff. Still. Made the morning a bit brighter.

That's life though, eh? Never the big things. Everyone thinks life's about weddings and holidays and birthdays. Nah. It's the daft little moments. Cup of tea that comes out just right. Robin landing on the fence. Having a laugh over something stupid. Finding the biscuits you forgot you'd bought. That's the good stuff. That's what sticks.

Anyway... Listen to me rambling. Old men do that. Start talking about bread. End up halfway through philosophy. Comes with the grey hair, I reckon. Or maybe we're just making sense a bit slower than we used to. Either way... Can't complain. Well... I could. But what's the point, eh?`;

const BRITISH_PERSON_SUMMARY = "Warm, reflective colloquial British English with a distinctly older, northern working-class voice. Uses short sentence fragments, pauses and ellipses, rhetorical questions, dry observational humour, gentle self-deprecation, and conversational asides. Characteristic wording includes ‘aye’, ‘lad’, ‘mind’, ‘now then’, ‘innit’, ‘costs nowt’, ‘daft’, ‘fella’, and ‘eh?’. Often begins with an everyday anecdote, wanders amiably through small details, and lands on a modest piece of homespun philosophy. Plain-spoken, good-natured, nostalgic without idealising the past, and attentive to small acts of human kindness.";

export const PREBUILT_PROFILES = Object.freeze([
  Object.freeze({
    id: "british-person",
    name: "British Person",
    summary: BRITISH_PERSON_SUMMARY,
    sourceText: BRITISH_PERSON_TEXT,
  }),
]);

export function getPrebuiltProfile(id) {
  const preset = PREBUILT_PROFILES.find((profile) => profile.id === id);
  if (!preset) return null;
  return {
    name: preset.name,
    samples: preset.sourceText.split(/\n\s*\n/).map((sample) => sample.trim()).filter(Boolean),
    style: {
      tone_description: "Warm, reflective, conversational, and gently nostalgic",
      humor_style: "Dry observational humour and affectionate self-deprecation",
      personality_notes: "Plain-spoken and good-natured; finds meaning in ordinary moments and small kindnesses",
      language_variety: "Colloquial British English with northern working-class phrasing",
      slang_and_regionalisms: ["aye", "lad", "mind", "now then", "innit", "costs nowt", "daft", "fella", "eh"],
    },
    summary: preset.summary,
    voiceId: null,
    requiresVerification: false,
    preferenceRules: [],
    builtInId: preset.id,
  };
}
