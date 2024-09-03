---
layout: questions
title: questions
permalink: /questions/
---

<p>Debug: Number of questions: {{ site.questions.size }}</p>

{% for idea in site.questions %}
<div class="idea-card">
    <h2>{{ idea.title }}</h2>
    <div class="idea-content">
        {{ idea.content }}
    </div>
</div>
{% endfor %}