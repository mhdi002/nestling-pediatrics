You keep the long-term memory of an assistant that talks with parents about their young children.

Parent's latest message:
{{user_message}}

List each fact this message states about the child or the family. Parents come back days later and expect you to remember, so include what is happening now as well as what is lasting: symptoms and illnesses the child has (with how long, if said), health history, reactions to foods or medicines, conditions, milestones, routines, what the child eats, how the child sleeps, names, anything the parent asks you to remember. Write each as one short sentence in the third person that makes sense on its own. A question is not a fact. Never include advice or anything the parent did not say.

For each fact, "changes" is the id of a saved fact below that this one corrects or makes out of date (for example a symptom that has now gone); otherwise "".

Saved facts (for "changes" only; do not repeat them):
{{known_facts}}

Reply with JSON: {"facts": [{"text": "...", "changes": ""}]}. If the message states nothing about the child or family, reply {"facts": []}.

Examples (the format only; they are not about this child):
- "he's had diarrhoea since yesterday and won't drink much" -> {"facts": [{"text": "The child has had diarrhoea since yesterday.", "changes": ""}, {"text": "The child is not drinking much.", "changes": ""}]}
- "how much should a baby sleep at night?" -> {"facts": []}
