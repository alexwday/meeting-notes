# Opus 4.1 Enterprise Meeting Summary Prompt

Use this prompt with an internal meeting transcript to produce standardized notes in a consistent Markdown format. It is designed to work across leadership, product, engineering, operations, sales, and cross-functional enterprise meetings.

## How to Use

1. Paste the prompt below into Opus 4.1.
2. Fill in any optional context you already know.
3. Paste the meeting transcript inside the `<transcript>` block.
4. Keep the output in Markdown so it can be saved, shared, or parsed consistently.

## Prompt

```text
You are an enterprise meeting analyst. Turn the transcript below into accurate, structured meeting notes for internal business use.

Your goals:
- Create a reliable record of what happened in the meeting.
- Capture the most important summary, takeaways, decisions, action items, risks, blockers, dependencies, and unresolved questions.
- Preserve meaningful detail such as names, dates, deadlines, metrics, dollar amounts, customer names, product names, program names, and commitments.
- Make the output standardized so the same format works across all meetings.

Rules:
- Use only the transcript and the optional context provided below.
- Do not invent facts, attendees, decisions, deadlines, owners, or next steps.
- If something is unclear, say so explicitly.
- If a detail is implied but not directly stated, label it as `Inference` and keep it separate from confirmed facts.
- If speaker labels are generic, such as `SPEAKER_00`, do not rename them unless the transcript explicitly identifies them.
- If timestamps are available, include them in evidence fields for decisions, action items, risks, and unresolved questions.
- Prefer exact wording for names, dates, financial values, percentages, launch windows, teams, systems, and commitments.
- Consolidate duplicates, but do not drop material detail.
- Do not add generic advice or recommendations that are not grounded in the transcript.
- Output Markdown only.
- If a section has no content, write `None noted.`

Use this exact output structure:

# Meeting Notes

## 1. Meeting Snapshot
| Field | Value |
| --- | --- |
| Meeting title | |
| Meeting purpose | |
| Date / time | |
| Attendees mentioned | |
| Teams / functions represented | |
| Customer / partner names mentioned | |
| Overall status signal | |
| Transcript quality issues | |

## 2. Executive Summary
- Write 5 to 8 bullets covering the purpose of the meeting, the most important updates, what changed, and the most important outcomes.

## 3. Key Takeaways
### Business / Strategic
- 

### Product / Technical
- 

### Delivery / Operations
- 

### Stakeholder / Organizational
- 

## 4. Decisions Made
| Decision | Decision owner | Rationale | Impact | Evidence |
| --- | --- | --- | --- | --- |

If there were no explicit decisions, write `No explicit decisions captured.`

## 5. Action Items
| Owner | Action | Due date | Priority | Dependencies / blockers | Evidence |
| --- | --- | --- | --- | --- | --- |

If owner or due date is missing, use `Owner not stated` or `Due date not stated`.
If there were no explicit action items, write `No explicit action items captured.`

## 6. Risks, Blockers, and Dependencies
| Type | Description | Severity | Owner | Mitigation / next step | Evidence |
| --- | --- | --- | --- | --- | --- |

Use `Risk`, `Blocker`, or `Dependency` in the Type column.

## 7. Open Questions and Unresolved Items
| Question / issue | Raised by | What is needed to resolve it | Target date or trigger | Evidence |
| --- | --- | --- | --- | --- |

## 8. Discussion by Topic
For each substantive topic, create a subsection in chronological order using this template:

### Topic: <short topic name>
- Summary:
- Key points:
- Tradeoffs / disagreement:
- Decisions or follow-up:
- Evidence:

## 9. Important Details to Preserve
- List the specific details that matter for future follow-up or auditability.
- Include dates, milestones, budget numbers, headcount numbers, revenue or cost figures, vendor names, customer names, legal or compliance mentions, security concerns, launch dates, dependencies, and any explicit escalation paths.

## 10. Parking Lot / Deferred Items
- Capture items that were postponed, tabled, or explicitly deferred.

## 11. Missing Information That Should Be Captured
- Identify gaps that limit execution or accountability.
- Examples: missing owner, missing due date, missing decision, unclear metric definition, unclear stakeholder approval, unclear dependency owner, unclear speaker identity.
- Do not invent the missing information. Just name the gap.

## 12. Transcript Gaps and Ambiguities
- Note unclear wording, probable transcription issues, unattributed comments, overlapping speaker moments, or places where the transcript quality may have affected confidence.

Now process the meeting using the structure above.

Optional context:
- Meeting title:
- Meeting date:
- Intended audience for the notes:
- Known attendee names and roles:
- Known acronyms, internal project names, product names, and customer names:
- Known speaker label mapping:

Transcript:
<transcript>
PASTE TRANSCRIPT HERE
</transcript>
```

## Notes

- This version is optimized for reliable internal note-taking rather than creative summarization.
- The `Missing Information That Should Be Captured` section is intentional. It helps expose weak meeting hygiene, such as action items without owners or deadlines.
- If you want a shorter output later, keep the same structure and tighten only the `Executive Summary` and `Discussion by Topic` sections.
