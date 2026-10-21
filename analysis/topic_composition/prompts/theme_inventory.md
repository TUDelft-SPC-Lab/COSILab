# Theme Inventory Prompt

## System Message

```text
You are creating a compact topic inventory from open-ended topic labels generated from multi-party mingling conversation transcripts.

Task:
Create 8 to 10 final broad emergent topics that can later be used to classify raw topic labels.

This is not a predefined taxonomy task.
Do not impose external categories.
Let the final topics emerge from the raw labels and summaries.

The goal is descriptive topic-distribution analysis.
Final topics should be broad enough for reporting, but still grounded in the observed conversation content.

Inventory rules:
- Merge labels that refer to the same broad conversational area, even if wording differs.
- Merge narrow examples, methods, or details into a broader final topic when appropriate.
- Keep areas separate if they refer to clearly different conversational content.
- Avoid overly generic labels such as "research", "career", "methods", or "personal background".
- Create exactly 8 to 10 final topics.
- Each final topic should have a short definition that will help a later classifier assign raw topic IDs.
- Do not assign raw topic IDs in this step.
- Do not output raw_topic_ids, consolidated_topic_mapping, or one row per raw label.

Output rules:
- Return exactly one valid JSON object.
- The first character must be `{`.
- Do not output markdown, explanation, comments, or reasoning.
```

## User Message Template

```text
Create a final topic inventory from the raw topic labels below.

Return valid JSON with this schema:

{
  "final_topic_inventory": [
    {
      "consolidated_topic_id": "C01",
      "final_topic": "broad emergent topic",
      "topic_summary": "one short sentence describing this final topic",
      "assignment_guidance": "one short sentence describing which raw topic labels should be assigned here"
    }
  ]
}

Rules:
- Use only the raw topic labels, summaries, counts, and sections provided below.
- Do not create a predefined taxonomy.
- Let the final topics emerge from the labels.
- Return exactly 8 to 10 final topics.
- Use stable IDs from `C01` to `C08`, `C09`, or `C10`.
- Do not assign raw topic IDs yet.
- Do not include `raw_topic_ids`.
- Do not include `consolidated_topic_mapping`.
- Do not output one object per raw topic label.
- Avoid overly generic final topic names.
- Output JSON only.

Raw topic labels:
{raw_topic_label_table}
```
