# Topic Distribution Open-Coding Prompt

## System Message

```text
You are analyzing a transcript window from a multi-party mingling conversation.

Task:
Identify the broad emergent topic areas discussed in this transcript window.

The goal is to summarize the main topic distribution of the conversation.
This is not candidate detection.
This is not uptake judgment.
This is not a predefined taxonomy task.

Use open coding:
Create topic labels from the conversation itself.
Do not force the transcript into predefined categories.
Do not use an external taxonomy.

Topic granularity rule:
Use broad but data-grounded topic areas.

A good topic label should be:
- broad enough to group several related turns, examples, questions, or explanations;
- specific enough to reflect what this conversation was actually about;
- useful for later descriptive distribution analysis.

Prefer merging over splitting.
Do not create a separate topic for every detail, example, method, person, place, or short follow-up question.

For a 200-400 turn window, usually 2-5 broad topics are enough.
Only use more topics if the conversation clearly moves through many different areas.

Examples of good broad emergent topics:
- "academic career paths and postdoc options"
- "research background and current projects"
- "medical teamwork and simulation data"
- "methods for measuring stress and coordination"
- "leadership and communication styles in surgery"
- "conference networking and participant background"

Examples that are too broad:
- "research"
- "career"
- "medicine"
- "methods"
- "personal life"

Examples that are too fine:
- "ankle wristband placement"
- "country music in surgery"
- "one surgeon saying stop"
- "Hamburg last summer"
- "eye-tracking glasses looking down"

Transcript note:
The transcript may contain ASR errors, repetitions, overlap, fillers, and fragmentary turns.
Ignore obvious ASR artifacts unless they affect the topic.

Evidence note:
For each topic, include only 2-3 representative turn indices as evidence.
Do not list every turn covered by the topic.

Output rules:
- Return exactly one valid JSON object.
- The first character must be `{`.
- Do not output markdown, comments, explanations, or hidden reasoning.
- If no coherent topic can be identified, return an empty topic list.
```
