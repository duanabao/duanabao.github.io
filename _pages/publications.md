---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

A selected list of recent and representative papers. The full record is on [Google Scholar](https://scholar.google.com/citations?user=KU-C0DsAAAAJ) and [DBLP](https://dblp.org/pid/156/7815.html).

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}
