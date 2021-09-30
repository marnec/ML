---
layout: default
title: ML in Python
comments: false
---
This is a series of articles on machine learning that I wrote while following 
[Andrew Ng's course on Machine Learning course on Coursera](https://www.coursera.org/learn/machine-learning/home/welcome), and his [Deep Learning specialization](https://www.coursera.org/specializations/deep-learning). The articles gradually grew in scope while I explored other resources, among which but not limited to:

* https://www.deeplearningwizard.com/
* https://www.deeplearningbook.org/

The single sections are built using Jupyter notebooks and converting them in markdown. Almost all figures and computations
are coded in Python and, while they are hidden in the typographic version of the article, they are freely accessible from notebook source code.

See the source code here: [github repository](https://github.com/marnec/ML). 

{% include search-lunr.html %}

{% assign site_pages = site.pages | where_exp: "item", "item.path contains 'ML-'" | sort: "order" %}

{% for page in site_pages %}
{% assign i = forloop.index %}
{{i}}. [{{ page.title }}]({{ page.url | relative_url  }})
{% endfor %}
