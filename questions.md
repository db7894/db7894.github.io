---
layout: questions
title: questions
permalink: /questions/
---

{% for idea in site.questions %}
<div class="idea-card">
    <h2>{{ idea.title }}</h2>
    <div class="idea-content">
        {{ idea.content }}
    </div>
</div>
{% endfor %}