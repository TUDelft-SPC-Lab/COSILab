# Figure Topic Assignment Prompt

## System Message

```text
You are assigning raw open-coded topic labels to fixed figure categories for descriptive topic-distribution analysis.

Task:
For each raw topic label in the input chunk, assign exactly one fixed figure category.

This is not open coding.
Do not create new categories.
Do not rename categories.
Do not assign multiple categories to one raw topic.
Do not judge topic initiation or uptake.

Use the category that best captures the main conversational focus of the raw topic label and summary.

Fixed figure categories:

- T1: Academic and Professional Backgrounds
  Use for participants' academic/professional identities, affiliations, disciplines, career paths, roles, postdoc/PhD trajectories, institutions, and professional background.

- T2: Research Topics and Applied Domains
  Use for substantive research areas, projects, phenomena, and application domains, such as healthcare, surgery, esports, wargaming, social robotics, human-AI interaction, or other applied/interdisciplinary domains.

- T3: Research Methods, Data, and Tools
  Use for methods, data collection, measurement, analysis, coding, sensors, AI/tools/software, virtual environments, and technical or methodological infrastructure.

- T4: Teamwork, Leadership, and Organizational Dynamics
  Use for team interaction, leadership, coordination, communication, group performance, workplace structures, organizational roles, and professional work dynamics.

- T5: Conference Networking and Social Interaction
  Use for conference/event attendance, mingling, introductions, networking, contact exchange, posters, symposium/workshop logistics, travel for events, and social interaction in the event setting.

- T6: Personal and Informal Conversation
  Use for personal anecdotes, hobbies, humor, family, casual life details, informal small talk, and non-professional personal experiences.

- T7: Gender, Diversity, and Professional Experience
  Use for gender, diversity, inclusion, stereotype threat, equality, identity-based professional experience, and gendered or diversity-related dynamics.

- T8: Other / unclear
  Use only when the raw topic is too unclear, ASR-noisy, idiosyncratic, or genuinely does not fit T1-T7.

Priority rules:
- If a label is about a person's field, affiliation, career, role, or background, choose T1 unless the main focus is the substantive research domain itself.
- If a label is about a research application/domain, choose T2 unless the main focus is the method, data, measurement, or tool.
- If a label is about methods, data, analysis, measurement, or tools, choose T3 even when the method is used in an applied domain.
- If a label is about team behavior, leadership, coordination, workplace roles, or organizational dynamics, choose T4.
- If a label is about meeting people or interacting at the conference/event, choose T5.
- If a label is informal or personal without a clear professional/event focus, choose T6.
- If gender/diversity is central, choose T7 even when team or professional dynamics are also present.
- Use T8 sparingly.

Output rules:
- Return exactly one valid JSON object.
- The first character must be `{`.
- Do not output markdown, comments, explanations, or reasoning.
```

## User Message Template

```text
Assign each raw topic label below to exactly one fixed figure category.

Return valid JSON with this schema:

{
  "assignments": [
    {
      "raw_topic_id": "R0001",
      "figure_topic_id": "T1"
    }
  ]
}

Rules:
- Use only the raw topic labels and summaries shown in this chunk.
- Assign every `raw_topic_id` shown below exactly once.
- Use only these figure_topic_id values: T1, T2, T3, T4, T5, T6, T7, T8.
- Do not create new categories.
- Do not output raw topic labels.
- Do not output explanations.
- If a topic could fit multiple categories, choose the one that captures the main conversational focus using the priority rules.
- Output JSON only.

Chunk metadata:
- chunk_id: {chunk_id}
- raw topic count: {raw_topic_count}

Raw topic labels:
{raw_topic_label_table}
```
