---
layout: default
title: "Recommendation systems"
categories: recommender
permalink: /ML25/
order: 25
comments: true
---

# Recommendation algorithms
Recommender systems are somehow ignored in academia but very hot in industry

Suppose you have a dataset like the one in the table below, with the number of users $n_u=4$ and the number of movies $n_m=5$. 




<style  type="text/css" >
</style><table id="T_75bb3_" ><caption>Table 1 - Ratings of 5 movies assigned by 4 users</caption><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Alice</th>        <th class="col_heading level0 col1" >Bob</th>        <th class="col_heading level0 col2" >Carol</th>        <th class="col_heading level0 col3" >Dave</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_75bb3_level0_row0" class="row_heading level0 row0" >Love at last</th>
                        <td id="T_75bb3_row0_col0" class="data row0 col0" >5.0</td>
                        <td id="T_75bb3_row0_col1" class="data row0 col1" >5.0</td>
                        <td id="T_75bb3_row0_col2" class="data row0 col2" >0.0</td>
                        <td id="T_75bb3_row0_col3" class="data row0 col3" >0.0</td>
            </tr>
            <tr>
                        <th id="T_75bb3_level0_row1" class="row_heading level0 row1" >Romance forever</th>
                        <td id="T_75bb3_row1_col0" class="data row1 col0" >5.0</td>
                        <td id="T_75bb3_row1_col1" class="data row1 col1" >nan</td>
                        <td id="T_75bb3_row1_col2" class="data row1 col2" >nan</td>
                        <td id="T_75bb3_row1_col3" class="data row1 col3" >0.0</td>
            </tr>
            <tr>
                        <th id="T_75bb3_level0_row2" class="row_heading level0 row2" >Cute puppies</th>
                        <td id="T_75bb3_row2_col0" class="data row2 col0" >nan</td>
                        <td id="T_75bb3_row2_col1" class="data row2 col1" >4.0</td>
                        <td id="T_75bb3_row2_col2" class="data row2 col2" >0.0</td>
                        <td id="T_75bb3_row2_col3" class="data row2 col3" >nan</td>
            </tr>
            <tr>
                        <th id="T_75bb3_level0_row3" class="row_heading level0 row3" >Car chases</th>
                        <td id="T_75bb3_row3_col0" class="data row3 col0" >0.0</td>
                        <td id="T_75bb3_row3_col1" class="data row3 col1" >0.0</td>
                        <td id="T_75bb3_row3_col2" class="data row3 col2" >5.0</td>
                        <td id="T_75bb3_row3_col3" class="data row3 col3" >4.0</td>
            </tr>
            <tr>
                        <th id="T_75bb3_level0_row4" class="row_heading level0 row4" >Katana</th>
                        <td id="T_75bb3_row4_col0" class="data row4 col0" >0.0</td>
                        <td id="T_75bb3_row4_col1" class="data row4 col1" >0.0</td>
                        <td id="T_75bb3_row4_col2" class="data row4 col2" >5.0</td>
                        <td id="T_75bb3_row4_col3" class="data row4 col3" >nan</td>
            </tr>
    </tbody></table>



Some users did not rated some movies. To denote if a movie $i$ has been rated by user $j$ we denote $r(i,j)=1$ if the movie is rated, $0$ otherwise. If defined, we denote $y^{(i, j)}$ as the rating assigned to movie $i$ by user $j$.

A recommendation system tries to predict the rating of unrated movies for each user based on ratings that he/she already assigned. 

## Content-based recommendation system
We can see that in our case Alice and Bob prefer romance movies, while Carol and Dave prefer action movies. 

It will be useful to define how much each movie belong to each of these categories, so we define two features that express how much each movies is aligned with the definition of "romance" ($x_1$) and "action" ($x_2$).




<style  type="text/css" >
</style><table id="T_85b35_" ><caption>Table 2 - Ratings of 5 movies assigned by 4 users and features indicating the level of romance ($x_1$) and action ($x_2$) of each movie</caption><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Alice</th>        <th class="col_heading level0 col1" >Bob</th>        <th class="col_heading level0 col2" >Carol</th>        <th class="col_heading level0 col3" >Dave</th>        <th class="col_heading level0 col4" >$x_1$</th>        <th class="col_heading level0 col5" >$x_2$</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_85b35_level0_row0" class="row_heading level0 row0" >Love at last</th>
                        <td id="T_85b35_row0_col0" class="data row0 col0" >5.0</td>
                        <td id="T_85b35_row0_col1" class="data row0 col1" >5.0</td>
                        <td id="T_85b35_row0_col2" class="data row0 col2" >0.0</td>
                        <td id="T_85b35_row0_col3" class="data row0 col3" >0.0</td>
                        <td id="T_85b35_row0_col4" class="data row0 col4" >0.9</td>
                        <td id="T_85b35_row0_col5" class="data row0 col5" >0.0</td>
            </tr>
            <tr>
                        <th id="T_85b35_level0_row1" class="row_heading level0 row1" >Romance forever</th>
                        <td id="T_85b35_row1_col0" class="data row1 col0" >5.0</td>
                        <td id="T_85b35_row1_col1" class="data row1 col1" >nan</td>
                        <td id="T_85b35_row1_col2" class="data row1 col2" >nan</td>
                        <td id="T_85b35_row1_col3" class="data row1 col3" >0.0</td>
                        <td id="T_85b35_row1_col4" class="data row1 col4" >0.1</td>
                        <td id="T_85b35_row1_col5" class="data row1 col5" >0.0</td>
            </tr>
            <tr>
                        <th id="T_85b35_level0_row2" class="row_heading level0 row2" >Cute puppies</th>
                        <td id="T_85b35_row2_col0" class="data row2 col0" >nan</td>
                        <td id="T_85b35_row2_col1" class="data row2 col1" >4.0</td>
                        <td id="T_85b35_row2_col2" class="data row2 col2" >0.0</td>
                        <td id="T_85b35_row2_col3" class="data row2 col3" >nan</td>
                        <td id="T_85b35_row2_col4" class="data row2 col4" >1.0</td>
                        <td id="T_85b35_row2_col5" class="data row2 col5" >0.0</td>
            </tr>
            <tr>
                        <th id="T_85b35_level0_row3" class="row_heading level0 row3" >Car chases</th>
                        <td id="T_85b35_row3_col0" class="data row3 col0" >0.0</td>
                        <td id="T_85b35_row3_col1" class="data row3 col1" >0.0</td>
                        <td id="T_85b35_row3_col2" class="data row3 col2" >5.0</td>
                        <td id="T_85b35_row3_col3" class="data row3 col3" >4.0</td>
                        <td id="T_85b35_row3_col4" class="data row3 col4" >0.1</td>
                        <td id="T_85b35_row3_col5" class="data row3 col5" >1.0</td>
            </tr>
            <tr>
                        <th id="T_85b35_level0_row4" class="row_heading level0 row4" >Katana</th>
                        <td id="T_85b35_row4_col0" class="data row4 col0" >0.0</td>
                        <td id="T_85b35_row4_col1" class="data row4 col1" >0.0</td>
                        <td id="T_85b35_row4_col2" class="data row4 col2" >5.0</td>
                        <td id="T_85b35_row4_col3" class="data row4 col3" >nan</td>
                        <td id="T_85b35_row4_col4" class="data row4 col4" >0.0</td>
                        <td id="T_85b35_row4_col5" class="data row4 col5" >0.9</td>
            </tr>
    </tbody></table>



Each movie is now represented by a feature vector $ \lbrace x_0, x_1, x_2 \rbrace$ where $x_0=1$ is the intercept.

One of the way we could proceed is by treating each separate user as a separate linear regression problem. For each user $j$, we learn a parameter vector $\theta^{(j)} \in \mathbb{R}^3$, to predict user $j$ rating of movie $i$ as $(\theta ^{(j)})^Tx{(i)}$ stars.

Suppose we have somehow obtained a parameter vector $\theta_1 = [0, 5, 0]$ for Alice, we will have its rating for the "cute puppies" movie as:

$$x^{(3)} = 
\begin{bmatrix}
1 \\ 0.99 \\ 0
\end{bmatrix} \qquad \theta_1 = \begin{bmatrix}0 \\ 5 \\ 0\end{bmatrix}
$$

$$
\begin{aligned}
\left ( \theta ^{(1)} \right )^T x^{(3)} & = 5 \cdot 0.99 \\
& = 4.95
\end{aligned}
$$

So to learn $\theta^{(j)}$

$$
\min_{\theta^{(j)}} \frac{1}{2m_j} \sum_{i:r(i, j)=1} \left ( \left( \theta^{(j)} \right) ^Tx^{(i)} -y^{(i,j)} \right)^2 + \frac{\lambda}{2m^{(j)}} \sum^n_{k=1}\left(\theta_k^{(j)}\right)^2
$$

where:

* $r(i,j) = 1$ if user $j$ has rated movie $i$, $0$ otherwise;
* $y^{(i,j)}$ is the rating by user $j$ on movie $i$ (if defined)
* $\theta^{(j)}$ is the parameter vector for user $j$
* $x^{(i)}$ is the feature vector for movie $i$
* $m^{(j)}$ is the number of movies rated by user $j$

Usually by convention the term $m^{(j)}$ is eliminated, and we remain with

$$
\min_{\theta^{(j)}} \frac{1}{2} \sum_{i:r(i, j)=1} \left ( \left( \theta^{(j)} \right) ^Tx^{(i)} -y^{(i,j)} \right)^2 + \frac{\lambda}{2} \sum^n_{k=1}\left(\theta_k^{(j)}\right)^2
$$

And to learn the parameters vectors for all users $\theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(n_u)}$

$$
\begin{equation}
\min_{\theta^{(1)}, \ldots, \theta^{(n_u)}} \frac{1}{2} \sum^{n_u}_{j=1} \sum_{i:r(i, j)=1} \left ( \left( \theta^{(j)} \right) ^Tx^{(i)} -y^{(i,j)} \right)^2 + \frac{\lambda}{2} \sum^{n_u}_{j=1} \sum^n_{k=1}\left(\theta_k^{(j)}\right)^2
\end{equation}
\label{eq:learnratings} \tag{1}
$$

With gradient descent update

$$
\theta_k^{(j)} :=
\begin{cases}
\theta_k^{(j)} - \alpha \sum_{i:r(i, j)=1} \left ( \left( \theta^{(j)} \right) ^Tx^{(i)} -y^{(i,j)} \right) x_k^{(i)} \qquad & \text{for } k=0 \\
\theta_k^{(j)} - \alpha \left ( \sum_{i:r(i, j)=1} \left ( \left( \theta^{(j)} \right) ^Tx^{(i)} -y^{(i,j)} \right) x_k^{(i)} + \lambda\theta_k^{(j)} \right) \qquad & \text{for } k \neq 0
\end {cases}
$$

## Collaborative filtering
Content-based approaches are based on the assumption that we know the features to use for movies and that those features are meaningful.

This is often not the case and collaborative filtering is an approach that allows the algorithm to **learn what features are important**.

### Predicting movie genre from preference
In Table 2 we had features $x_1, x_2$ for each movie, this means that someone has evaluated and assigned those values manually. This is often not the case so let's imagine a different situation where we don't have $x_1, x_2$




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alice</th>
      <th>Bob</th>
      <th>Carol</th>
      <th>Dave</th>
      <th>$x_1$</th>
      <th>$x_2$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Love at last</th>
      <td>5.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Romance forever</th>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Cute puppies</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Car chases</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Katana</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



but we know what genre each user prefers

$$
\theta^{(1)} = 
\begin{bmatrix}
0 \\ 5 \\ 0
\end{bmatrix} 
\qquad 
\theta^{(2)} = 
\begin{bmatrix}
0 \\ 5 \\ 0
\end{bmatrix} 
\qquad 
\theta^{(3)} = 
\begin{bmatrix}
0 \\ 0 \\ 5
\end{bmatrix} 
\qquad 
\theta^{(4)} = 
\begin{bmatrix}
0 \\ 0 \\ 5
\end{bmatrix} 
$$

With this information it becomes possible to infer $x_1, x_2$ for each movie. Given $\theta^{(1)}, \ldots, \theta^{(n_u)}$ to learn $x^{(i)}$

$$
\min_{x^{(i)}} \frac{1}{2} \sum_{j:r(i, j)=1} \left ( \left( \theta^{(j)} \right) ^Tx^{(i)} -y^{(i,j)} \right)^2 + \frac{\lambda}{2} \sum^n_{k=1}\left(x_k^{(i)}\right)^2
$$

Given $\theta^{(1)}, \ldots, \theta^{(n_u)}$ to learn $x^{(1)}, \ldots, x^{(n_m)}$

$$
\begin{equation}
J= \frac{1}{2} \sum^{n_m}_{i=1} \sum_{j:r(i, j)=1} \left ( \left( \theta^{(j)} \right) ^Tx^{(i)} -y^{(i,j)} \right)^2 + \frac{\lambda}{2} \sum^{n_m}_{i=1} \sum^n_{k=1}\left(x_k^{(i)}\right)^2
\end{equation}
\label{eq:learnpreference} \tag{2}
$$

### Collaborative filtering
So we have established that:

* Given $x^{(1)}, \ldots, x^{(n_m)}$ we can estimate $\theta^{(1)}, \ldots, \theta^{(n_u)}$ with $\eqref{eq:learnratings}$ 
* Given $\theta^{(1)}, \ldots, \theta^{(n_u)}$ we can estimate $x^{(1)}, \ldots, x^{(n_m)}$ with $\eqref{eq:learnpreference}$

By randomly initializing $\theta$ we can iteratively obtain better and better estimates of both $\theta$ and $x$.

However, there is a more efficient algorithm that doesn't need to go back and forth between minimizing respect to $\theta$ and $x$ and it is called **collaborative filtering**.

Collaborative filtering is based on the combination of $\eqref{eq:learnratings}$ and $\eqref{eq:learnpreference}$, where the cost function $J$ is defined as

$$
\begin{aligned}
J(x^{(1)}, \ldots, x^{(n_m)},\theta^{(1)}, \ldots, \theta^{(n_u)}) & = \frac{1}{2} \sum_{(i,j):r(i,j)=1} \left ( \left( \theta^{(j)} \right) ^Tx^{(i)} -y^{(i,j)} \right)^2 \\
&+ \frac{\lambda}{2} \sum^{n_m}_{i=1} \sum^n_{k=1}\left(x_k^{(i)}\right)^2\\
&+ \frac{\lambda}{2} \sum^{n_u}_{j=1} \sum^n_{k=1}\left(\theta_k^{(j)}\right)^2
\end{aligned}
$$

and the minimization objective becomes

$$
\min_{x^{(1)}, \ldots, x^{(n_m)},\theta^{(1)}, \ldots, \theta^{(n_u)}} J(x^{(1)}, \ldots, x^{(n_m)},\theta^{(1)}, \ldots, \theta^{(n_u)}) 
$$

Do notice that the convention $x_0=1$ is not adopted here and consequently we do not have $\theta_0$. So we have $x \in \mathbb{R}^n$ and $\theta \in \mathbb{R}^n$. The reason behind this choice is that since the algorithm is learning all the features, if it needs a feature to be $1$ it can (and will) set it itself.

So the training procedure will take the following steps:

1. In order for the training procedure to work we need to initialize $x$ and $\theta$ to some small random values, similarly to neural network training. This serves as symmetry breaking and ensures the algorithm learns features $x^{(1)}, \dots, x^{(n_m)}$ that are different from each other.

2. Then we minimize $J(x^{(1)}, \ldots, x^{(n_m)},\theta^{(1)}, \ldots, \theta^{(n_u)}) $ using gradient descent (or an optimization algorithm). For each $j=1, \ldots, n_u, i=1, \ldots, n_m$

$$
\begin{aligned}
x^{(i)}_k & := x^{(i)}_k - \alpha \left ( \sum_{j:r(i, j)=1} \left ( \left( \theta^{(j)} \right)^T x^{(i)} - y^{(i,j)} \right) \theta_k^{(j)} + \lambda x_k^{(i)} \right) \\
\theta_k^{(j)} & := \theta_k^{(j)} - \alpha \left ( \sum_{i:r(i, j)=1} \left ( \left( \theta^{(j)} \right) ^Tx^{(i)} -y^{(i,j)} \right) x_k^{(i)} + \lambda\theta_k^{(j)} \right)
\end{aligned}
$$

3. For a user with parameters $\theta$ and a movie with learned features $x$, predict a rating $\theta^Tx$

## Finding related movies
For each product $i$ we learn a feature vector $x^{(i)} \in \mathbb{R}^n$. In order to find movies $j$ related to movie $i$ we have to find small values of the distance between the two movies 

$$\| x^{(i)} - x^{(j)} \|$$
