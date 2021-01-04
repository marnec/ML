---
layout: default
title: Notes on Machine Learning
---
These websites contains notes on machine learning that I wrote while following the
[Professor Ng's course on Machine Learning course on Coursera](https://www.coursera.org/learn/machine-learning/home/welcome)

The whole notes are built using Jupyter notebooks, the figures examples and calculations are done using Python.
For readibility purposes most of the source code is hidden. If you wish to see the source code you can download the
jupyter notebooks from the repository.

{% assign site_pages = site.pages | where_exp: "item", "item.path contains 'ML-'" | sort: "order" %}

{% for page in site_pages %}
{% assign i = forloop.index %}
{{i}}. [{{ page.title }}]({{ page.url | relative_url  }})
{% endfor %}
