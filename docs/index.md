---
layout: default
title: Notes on Machine Learning
---
These websites contains notes on machine learning that I wrote while following the
[Professor Ng's course on Machine Learning course on Coursera](https://www.coursera.org/learn/machine-learning/home/welcome)
[ML1]({{ site.baseurl }}{% link /ML1 %})

{% for page in site.pages %}
  {% if page.url == '/path/to/page.html' %}
[{{ page.title }}]({{ page.url }})
  {% endif %}
{% endfor %}
