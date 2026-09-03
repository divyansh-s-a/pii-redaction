# End-to-end cost per configuration

Generated 2026-09-02 by `tools/cost_report.py` | corpus: 1.17 audio-hours | OpenRouter rows measured/estimated: 145/0

| config | transcription $ | detection $ | total $ | $/audio-hour |
|---|---|---|---|---|
| ElevenLabs + Gemini (production) | 0.31 | 0.51 | 0.81 | 0.70 |
| Gemini end-to-end | 18.50 | 0.51 | 19.01 | 16.26 |
| Whisper + Gemini | 0.50 | 0.33 | 0.84 | 0.71 |
| ElevenLabs + GPT detector | 0.31 | 0.03 | 0.34 | 0.29 |
| Whisper + GPT detector | 0.50 | 0.02 | 0.52 | 0.45 |

Safety-gate overhead (production deliveries; applies to whichever config ships): $0.85 total, $0.73 per audio-hour

| gate | $ |
|---|---|
| audit | 0.54 |
| edge_verify | 0.28 |
| judge_other | 1.07 |
| pverify | 0.04 |

Prices: ElevenLabs $0.22/hr (published rate x measured seconds), Whisper $0.36/hr (measured), Gemini Pro $2.00/$12.00 per 1M in/out+thinking (derived; reproduces the v1 measured sample), Flash $0.30/$2.50 per 1M (estimate), OpenRouter judge/detector rows measured via usage accounting where available.
