---
layout: default
title: "Deep Learning - RNN - Sequence to sequence models"
categories: deeplearning
permalink: /ML46/
order: 46
comments: true
---

# Sequence to sequence models
Sequence to sequence models are RNN algorithms that take sequences as input and produce sequences as output. They are employed in many fields from machine translation to speech recognition.

Machine translation is the perfect example of the reason why sequence to sequence models have been developed. Almost all the examples seen so far map an input sequence to a single output. In machine translation we want to translate a sentence from a language to a second sequence in another. 

## Basic seq-to-seq model
Suppose we have a sentence in French that we want to map to its English translation:

$$
\begin{split}
&\begin{array}
&x^{\langle 1 \rangle} & x^{\langle 2 \rangle} & x^{\langle 3 \rangle} & x^{\langle 4 \rangle} & x^{\langle 5 \rangle} \\
\text{Jane} & \text{visite} & \text{l'Afrique} & \text{en} & \text{septembre}
\end{array}\\ \\
&\begin{array}
&\text{Jane} & \text{is} & \text{visiting} & \text{Africa} & \text{in} & \text{September} \\
y^{\langle 1 \rangle} & y^{\langle 2 \rangle} & y^{\langle 3 \rangle} & y^{\langle 4 \rangle} & y^{\langle 5 \rangle} & y^{\langle 6 \rangle} 
\end{array}
\end{split}
$$

In order to map such input sequences to output sequences, [Sutskever et.al. 2014](https://arxiv.org/abs/1409.3215) and [Cho et.al.2014](https://arxiv.org/abs/1406.1078) developed RNN based seq-to-seq models.


```python
f = Flow(bbox=dict(boxstyle='square'))
for i in range(1, 9):
    d=1
    lbl = i if i < 5 else 'T_x'

    if i not in [2,7]:
        if i == 4:
            d=2
        f.node(f'a{i}', label='\n'*5+' '*3, fontsize=13, startpoint=f'a{i-1}', distance=d)
    else:
        f.node(f'a{i}', label='$\\cdots$', startpoint=f'a{i-1}', fontsize=13, bbox=dict(ec='none'))

    if i < 4 and i != 2:
        if i == 3:
            lbl = 'T_x'
        f.node(f'x{i}', label=f'$x^{{\\langle {lbl} \\rangle}}$', startpoint=f'a{i}', travel='s', fontsize=13, 
               edge_kwargs=dict(arrowprops=dict(arrowstyle='->')), bbox=dict(ec='none')) 

    if i >= 4 and i != 7:
        if i == 8:
            lbl = 'T_y'
        else:
            lbl = i - 3

        f.node(f'y{i}', label=f'$\\hat{{y}}^{{\\langle {lbl} \\rangle}}$', startpoint=f'a{i}', travel='n', fontsize=13,
               bbox=dict(ec='none'))
    
    if i > 4 and i < 7:
        f.edge(f'y{i-1}', f'a{i}', headport='s', tailport='s', 
           arrowprops=dict(connectionstyle='arc,angleA=-110,angleB=-10,armA=30,armB=10,rad=10', 
                           shrinkA=10, shrinkB=10, ec='gray'))
    if i == 8:
        f.edge(f'a{i-1}', f'a{i}', headport='s', tailport='s', 
           arrowprops=dict(connectionstyle='arc,angleA=-110,angleB=-10,armA=30,armB=10,rad=10', 
                           shrinkA=10, shrinkB=10, ec='gray', ls='--'))

f.node(startpoint='a1', label='$a^{\\langle 0 \\rangle}$', travel='w', fontsize=13, bbox=dict(ec='none'),
       edge_kwargs=dict(arrowprops=dict(arrowstyle='->')))

encoder_bbox = f.nodes['a2'].annotation.get_bbox_patch()
decoder_bbox = f.nodes['a6'].annotation.get_bbox_patch()
plt.annotate('encoder', (0.5, 3.5), (0.5, 3.8) , xycoords=encoder_bbox, textcoords=encoder_bbox, ha='center', 
             arrowprops=dict(arrowstyle='-[,widthB=5,lengthB=0.3,angleB=0', ec='r'))
plt.annotate('decoder', (0.5, -0.4), (0.5, -0.43) , xycoords=decoder_bbox, textcoords=decoder_bbox, ha='center', va='top',
             arrowprops=dict(arrowstyle='-[,widthB=9,lengthB=0.3,angleB=0', ec='r'));
```


    
![svg](ML-46-DeepLearningRNN4_files/ML-46-DeepLearningRNN4_2_0.svg)
    

