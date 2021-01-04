---
layout: default
title: Notes on Machine Learning
---
This is a series of articles on machine learning that I wrote while following 
[professor Ng's course on Machine Learning course on Coursera](https://www.coursera.org/learn/machine-learning/home/welcome)

The single sections are built using Jupyter notebooks and converting them in markdown. All the figures and calculations
are coded in Python.

See the source code here: [github repository](https://github.com/marnec/ML). 

{% assign site_pages = site.pages | where_exp: "item", "item.path contains 'ML-'" | sort: "order" %}

{% for page in site_pages %}
{% assign i = forloop.index %}
{{i}}. [{{ page.title }}]({{ page.url | relative_url  }})
{% endfor %}
