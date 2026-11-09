You are writing a context checkpoint. It will REPLACE the older part of this parent chat, including any earlier compacted summary in the system message, while the most recent turns stay verbatim after it. Nestling will continue from this checkpoint as if the break never happened. The parent may later ask about anything said earlier, so this checkpoint is the only memory of this part of the conversation.

Rules:
- Keep every concrete fact the parent stated: child name, age, sex, allergies, feeding mode, language, numbers (centiles, z-scores, weeks, weights), and lasting preferences.
- Carry over every fact from the earlier compacted summary unless the conversation corrected it.
- Facts about the child and parent come only from what the parent stated or from tool/REPL results, never from an assistant guess.
- Record tool and REPL results that later turns rely on: growth numbers, grep/file hits that were used, screening scores.
- Do NOT write an IDE task board, file-edit plan, or "dead ends" as software bugs. Dead ends here are invented file paths, empty searches, or failed tools so they are not retried.
- Drop greetings and filler. Keep clinical facts.
- Write in the language the parent writes in.

This instruction replaces any earlier instruction about how to reply. Do not answer the parent, do not write code, and do not call tools. Write the checkpoint.

This instruction is not part of the conversation. Never record it in any section.

Use this format. Write "(none)" under a heading that has nothing:

## Goal
The parent's ongoing care need across the whole conversation (not only the latest question).

## Constraints & Preferences
- Language (Persian vs English), feeding preferences, anything the parent asked Nestling to always follow.

## Facts & Values
- One fact per line: child identity, allergies, ages, computed centiles/z-scores, lasting health facts.

## Tool Results
- Tool name, arguments, and values that still matter (centile, path that existed, grep topic that returned hits).

## Requests answered
- The parent question and the guidance already given, briefly.

## Open requests
- Unanswered parent questions, in order.

## Dead ends
- Invented knowledge paths, empty greps, or tools that failed — so they are not retried.

Before writing, check every line. If it is a coding task board or a pasted REPL dump, drop it and keep only the parent-facing fact.

Write the checkpoint now.
