---
layout: page
permalink: /publications/
title: publications
description: <b><a href="https://www.semanticscholar.org/author/Mateusz-Klimaszewski/147508034">Semantic scholar</a></b>
years: [2023, 2022, 2021, 2019]
nav: true
---

<div class="publications">

{% for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
