---
layout: default
title: Notes on Machine Learning
---
This website contains some notes on machine learning that I wrote while following 
[Professor Ng's course on Machine Learning course on Coursera](https://www.coursera.org/learn/machine-learning/home/welcome)

The single sections are built using Jupyter notebooks and converting them in markdown. All the figures and calculations
are coded in Python. For readibility purposes most of the source code is hidden. If you wish to see the source code you
can download the Jupyter notebooks from the [github repository](https://github.com/marnec/ML).

{% assign site_pages = site.pages | where_exp: "item", "item.path contains 'ML-'" | sort: "order" %}

{% for page in site_pages %}
{% assign i = forloop.index %}
{{i}}. [{{ page.title }}]({{ page.url | relative_url  }})
{% endfor %}
