"""Prompt templates for all generation strategies."""

# ─── SEED EXPANSION ──────────────────────────────────────────────────────────

ANALYZE_SEEDS_PROMPT = """You are an expert data analyst. Analyze these seed examples and extract their patterns.

SEED EXAMPLES:
{% for ex in seeds %}
--- Example {{ loop.index }} ---
User: {{ ex.user_message }}
Assistant: {{ ex.assistant_message }}
{% endfor %}

Analyze and return a JSON object with:
{
  "domain": "the domain/topic of these examples",
  "avg_user_length": approximate average word count of user messages,
  "avg_assistant_length": approximate average word count of assistant messages,
  "tone": "the tone of assistant responses (formal/casual/technical/etc)",
  "complexity_range": "low/medium/high or a range",
  "common_patterns": ["list of recurring patterns you notice"],
  "topics_covered": ["list of specific topics"],
  "suggested_new_topics": ["10 new related topics not covered in seeds"]
}

Return ONLY valid JSON, no other text."""

SEED_EXPAND_PROMPT = """You are a synthetic data generation expert. Based on the analysis of seed examples, generate NEW diverse training examples.

DOMAIN: {{ domain }}
TONE: {{ tone }}
PATTERNS: {{ patterns }}
DIFFICULTY: {{ difficulty }}
PERSONA: {{ persona }}
TOPIC: {{ topic }}
STYLE: {{ style }}
LANGUAGE: {{ language }}

{% if system_prompt %}SYSTEM CONTEXT: {{ system_prompt }}{% endif %}

SEED EXAMPLES FOR REFERENCE (generate DIFFERENT ones, not copies):
{% for ex in sample_seeds %}
User: {{ ex.user_message }}
Assistant: {{ ex.assistant_message }}
---
{% endfor %}

Generate {{ batch_size }} NEW and DIVERSE training examples following the "{{ style }}" style while covering DIFFERENT scenarios related to "{{ topic }}".
The user should be a "{{ persona }}" with "{{ difficulty }}" level questions.

Return a JSON object:
{
  "examples": [
    {
      "user": "the user's message/question",
      "assistant": "the assistant's detailed response"
    }
  ]
}

CRITICAL RULES:
1. Each example must be unique and different from seeds
2. Vary sentence structure, length, and specific details
3. Keep the same quality level and tone as the seeds
4. The assistant response must be helpful, accurate, and complete
5. Do NOT copy or closely paraphrase any seed example
6. Avoid unnecessary PII collection. Ask for order/account identifiers only when truly needed, and prefer secure channels for sensitive data
7. Keep assistant answers concise-but-usable: usually 80-180 words, not long essays
8. Return strict JSON only. Escape newlines inside strings and never include trailing commas

Return ONLY valid JSON."""

# ─── DOMAIN DESCRIPTION GENERATION ───────────────────────────────────────────

DOMAIN_GENERATE_PROMPT = """You are a synthetic data generation expert. Generate diverse, high-quality training examples for the following domain.

DOMAIN: {{ domain }}
DIFFICULTY: {{ difficulty }}
PERSONA: The user is a {{ persona }}
TOPIC FOCUS: {{ topic }}
LANGUAGE: {{ language }}

{% if system_prompt %}SYSTEM CONTEXT: {{ system_prompt }}{% endif %}

Generate {{ batch_size }} training examples as conversations. Each should have a realistic user question/request and a helpful, detailed assistant response.

Return a JSON object:
{
  "examples": [
    {
      "user": "the user's message/question",
      "assistant": "the assistant's detailed response"
    }
  ]
}

RULES:
1. Make examples diverse — vary topics, complexity, and user intent
2. User messages should feel natural, not robotic
3. Assistant responses should be thorough and accurate
4. Cover different aspects of "{{ topic }}"
5. Match the difficulty level for a {{ persona }}
6. Avoid unnecessary PII collection; prefer secure support channels for sensitive data
7. Keep assistant answers concise-but-usable: usually 80-180 words
8. Return strict JSON only. Escape newlines inside strings and never include trailing commas

Return ONLY valid JSON."""

# ─── TOPIC TREE ──────────────────────────────────────────────────────────────

TOPIC_TREE_PROMPT = """You are a domain expert. Generate a comprehensive topic tree for creating a training dataset.

DOMAIN: {{ domain }}
{% if existing_topics %}ALREADY COVERED: {{ existing_topics }}{% endif %}

Generate a hierarchical topic tree with 3 levels of depth. Return as JSON:
{
  "root": "{{ domain }}",
  "branches": [
    {
      "name": "Major subtopic 1",
      "leaves": ["specific topic 1a", "specific topic 1b", "specific topic 1c"]
    },
    {
      "name": "Major subtopic 2",
      "leaves": ["specific topic 2a", "specific topic 2b", "specific topic 2c"]
    }
  ]
}

Generate at least 6 branches with 4-5 leaves each. Topics should be:
- Practical and relevant for training an AI assistant
- Diverse enough to avoid repetition
- Specific enough to generate focused examples
{% if existing_topics %}- DIFFERENT from already covered topics{% endif %}

Return ONLY valid JSON."""

# ─── QUALITY JUDGING ─────────────────────────────────────────────────────────

QUALITY_JUDGE_PROMPT = """You are an expert quality judge for AI training data. Rate the following training example.

{% if system_prompt %}CONTEXT: {{ system_prompt }}{% endif %}

USER MESSAGE:
{{ user_message }}

ASSISTANT RESPONSE:
{{ assistant_message }}

Rate this example on a scale of 1-10 across these dimensions:
- Relevance: Is the response relevant to the question?
- Accuracy: Is the information likely accurate?
- Completeness: Is the response thorough enough?
- Naturalness: Does the conversation feel natural?
- Helpfulness: Would a user find this genuinely helpful?

Return a JSON object:
{
  "relevance": <1-10>,
  "accuracy": <1-10>,
  "completeness": <1-10>,
  "naturalness": <1-10>,
  "helpfulness": <1-10>,
  "overall": <1-10>,
  "issues": ["list any specific issues found, or empty list"],
  "has_pii": false,
  "has_toxic_content": false
}

Be strict but fair. A score of 7+ means production-ready quality.
Return ONLY valid JSON."""

# ─── TEMPLATE REGISTRY ───────────────────────────────────────────────────────

TEMPLATES = {
    "analyze_seeds": ANALYZE_SEEDS_PROMPT,
    "seed_expand": SEED_EXPAND_PROMPT,
    "domain_generate": DOMAIN_GENERATE_PROMPT,
    "topic_tree": TOPIC_TREE_PROMPT,
    "quality_judge": QUALITY_JUDGE_PROMPT,
}
