---
layout: default
title: "Neural Networks - Applications"
categories: neuralNetwork
permalink: /ML14/
order: 14
comments: true
---

# How neural networks build complex non-linear functions
In this section we will explain how a neural network can build relatively complex non-linear functions.

Let's take a non-linear classification example like that depicted below, where panel A is just a simplified version of panel B.


![png](ML-14-NeuralNetworksApplications_files/ML-14-NeuralNetworksApplications_2_0.png)


This is a non-linear classification example modeled by the logical XNOR function

$$
\begin{align}
&x_1 \; \text{XNOR} \; x_2 \\
&\text{NOT} \; (x_1 \; \text{XOR} \; x_2)
\end{align}
$$


## Logical AND function
Let's look at a neural network that can calculate the logical $\text{AND}$ function.

$$
\begin{align}
&x_1,x_2\in \lbrace 0,1 \rbrace\\
&y= x_1 \wedge x_2\\
&\Theta^{(1)} = 
\begin{bmatrix}-30\\20\\20\end{bmatrix}
\end{align}
$$


![png](ML-14-NeuralNetworksApplications_files/ML-14-NeuralNetworksApplications_5_0.png)


So that 

$$
\begin{equation}
h_\Theta(x) = g(-30+20x_1+20x_2)
\end{equation}
\label{eq:h} \tag{1}
$$

Since the sigmoid activation function $g(z)$ is


![png](ML-14-NeuralNetworksApplications_files/ML-14-NeuralNetworksApplications_7_0.png)


The the output of $\eqref{eq:h}$ is




<style  type="text/css" >
#T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row0_col0,#T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row0_col1,#T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row0_col2,#T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row1_col0,#T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row1_col1,#T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row1_col2,#T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row2_col0,#T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row2_col1,#T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row2_col2,#T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row3_col0,#T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row3_col1,#T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row3_col2{
            text-align:  left;
        }</style><table id="T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3" ><thead>    <tr>        <th class="col_heading level0 col0" >$x_1$</th>        <th class="col_heading level0 col1" >$x_2$</th>        <th class="col_heading level0 col2" >$h_\Theta(x)$</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row0_col0" class="data row0 col0" >0</td>
                        <td id="T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row0_col1" class="data row0 col1" >0</td>
                        <td id="T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row0_col2" class="data row0 col2" >$g(-30) \approx 0$</td>
            </tr>
            <tr>
                                <td id="T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row1_col0" class="data row1 col0" >0</td>
                        <td id="T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row1_col1" class="data row1 col1" >1</td>
                        <td id="T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row1_col2" class="data row1 col2" >$g(-10) \approx 0$</td>
            </tr>
            <tr>
                                <td id="T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row2_col0" class="data row2 col0" >1</td>
                        <td id="T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row2_col1" class="data row2 col1" >0</td>
                        <td id="T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row2_col2" class="data row2 col2" >$g(-10) \approx 0$</td>
            </tr>
            <tr>
                                <td id="T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row3_col0" class="data row3 col0" >1</td>
                        <td id="T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row3_col1" class="data row3 col1" >1</td>
                        <td id="T_456fb800_54dd_11eb_8e7f_40a3cc65d4e3row3_col2" class="data row3 col2" >$g(10) \approx 1$</td>
            </tr>
    </tbody></table>



Which is exactly the $x_1 \wedge x_2$.

## Logical OR function
The following network and table show instead $x_1 \vee x_2$


![png](ML-14-NeuralNetworksApplications_files/ML-14-NeuralNetworksApplications_11_0.png)


$$h_\Theta(x) = g(-10+20x_1+20x_2)$$




<style  type="text/css" >
#T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row0_col0,#T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row0_col1,#T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row0_col2,#T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row1_col0,#T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row1_col1,#T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row1_col2,#T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row2_col0,#T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row2_col1,#T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row2_col2,#T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row3_col0,#T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row3_col1,#T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row3_col2{
            text-align:  left;
        }</style><table id="T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3" ><thead>    <tr>        <th class="col_heading level0 col0" >$x_1$</th>        <th class="col_heading level0 col1" >$x_2$</th>        <th class="col_heading level0 col2" >$h_\Theta(x)$</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row0_col0" class="data row0 col0" >0</td>
                        <td id="T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row0_col1" class="data row0 col1" >0</td>
                        <td id="T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row0_col2" class="data row0 col2" >$g(-10) \approx 0$</td>
            </tr>
            <tr>
                                <td id="T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row1_col0" class="data row1 col0" >0</td>
                        <td id="T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row1_col1" class="data row1 col1" >1</td>
                        <td id="T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row1_col2" class="data row1 col2" >$g(10) \approx 1$</td>
            </tr>
            <tr>
                                <td id="T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row2_col0" class="data row2 col0" >1</td>
                        <td id="T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row2_col1" class="data row2 col1" >0</td>
                        <td id="T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row2_col2" class="data row2 col2" >$g(10) \approx 1$</td>
            </tr>
            <tr>
                                <td id="T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row3_col0" class="data row3 col0" >1</td>
                        <td id="T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row3_col1" class="data row3 col1" >1</td>
                        <td id="T_82324e42_54dd_11eb_8e7f_40a3cc65d4e3row3_col2" class="data row3 col2" >$g(30) \approx 1$</td>
            </tr>
    </tbody></table>



## Logical NOT function


![png](ML-14-NeuralNetworksApplications_files/ML-14-NeuralNetworksApplications_15_0.png)





<style  type="text/css" >
#T_8e704f56_54dd_11eb_8e7f_40a3cc65d4e3row0_col0,#T_8e704f56_54dd_11eb_8e7f_40a3cc65d4e3row0_col1,#T_8e704f56_54dd_11eb_8e7f_40a3cc65d4e3row1_col0,#T_8e704f56_54dd_11eb_8e7f_40a3cc65d4e3row1_col1{
            text-align:  left;
        }</style><table id="T_8e704f56_54dd_11eb_8e7f_40a3cc65d4e3" ><thead>    <tr>        <th class="col_heading level0 col0" >$x_1$</th>        <th class="col_heading level0 col1" >$h_\Theta(x)$</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_8e704f56_54dd_11eb_8e7f_40a3cc65d4e3row0_col0" class="data row0 col0" >0</td>
                        <td id="T_8e704f56_54dd_11eb_8e7f_40a3cc65d4e3row0_col1" class="data row0 col1" >$g(10) \approx 1$</td>
            </tr>
            <tr>
                                <td id="T_8e704f56_54dd_11eb_8e7f_40a3cc65d4e3row1_col0" class="data row1 col0" >1</td>
                        <td id="T_8e704f56_54dd_11eb_8e7f_40a3cc65d4e3row1_col1" class="data row1 col1" >$g(-10) \approx 0$</td>
            </tr>
    </tbody></table>



## Logical NOT-1 AND NOT-2 function


![png](ML-14-NeuralNetworksApplications_files/ML-14-NeuralNetworksApplications_18_0.png)





<style  type="text/css" >
#T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row0_col0,#T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row0_col1,#T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row0_col2,#T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row1_col0,#T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row1_col1,#T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row1_col2,#T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row2_col0,#T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row2_col1,#T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row2_col2,#T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row3_col0,#T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row3_col1,#T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row3_col2{
            text-align:  left;
        }</style><table id="T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3" ><thead>    <tr>        <th class="col_heading level0 col0" >$x_1$</th>        <th class="col_heading level0 col1" >$x_2$</th>        <th class="col_heading level0 col2" >$h_\Theta(x)$</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row0_col0" class="data row0 col0" >0</td>
                        <td id="T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row0_col1" class="data row0 col1" >0</td>
                        <td id="T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row0_col2" class="data row0 col2" >$g(10) \approx 1$</td>
            </tr>
            <tr>
                                <td id="T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row1_col0" class="data row1 col0" >0</td>
                        <td id="T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row1_col1" class="data row1 col1" >1</td>
                        <td id="T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row1_col2" class="data row1 col2" >$g(-10) \approx 0$</td>
            </tr>
            <tr>
                                <td id="T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row2_col0" class="data row2 col0" >1</td>
                        <td id="T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row2_col1" class="data row2 col1" >0</td>
                        <td id="T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row2_col2" class="data row2 col2" >$g(-10) \approx 0$</td>
            </tr>
            <tr>
                                <td id="T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row3_col0" class="data row3 col0" >1</td>
                        <td id="T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row3_col1" class="data row3 col1" >1</td>
                        <td id="T_aab5940a_54dd_11eb_8e7f_40a3cc65d4e3row3_col2" class="data row3 col2" >$g(-30) \approx 0$</td>
            </tr>
    </tbody></table>



## Logical XNOR function


![png](ML-14-NeuralNetworksApplications_files/ML-14-NeuralNetworksApplications_21_0.png)



![png](ML-14-NeuralNetworksApplications_files/ML-14-NeuralNetworksApplications_22_0.png)





<style  type="text/css" >
#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row0_col0,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row0_col1,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row0_col2,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row0_col3,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row0_col4,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row1_col0,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row1_col1,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row1_col2,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row1_col3,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row1_col4,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row2_col0,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row2_col1,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row2_col2,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row2_col3,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row2_col4,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row3_col0,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row3_col1,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row3_col2,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row3_col3,#T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row3_col4{
            text-align:  left;
        }</style><table id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3" ><thead>    <tr>        <th class="col_heading level0 col0" >$x_1$</th>        <th class="col_heading level0 col1" >$x_2$</th>        <th class="col_heading level0 col2" >$a_1^{(2)}$</th>        <th class="col_heading level0 col3" >$a_2^{(2)}$</th>        <th class="col_heading level0 col4" >$h_\Theta(x)$</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row0_col0" class="data row0 col0" >0</td>
                        <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row0_col1" class="data row0 col1" >0</td>
                        <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row0_col2" class="data row0 col2" >0</td>
                        <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row0_col3" class="data row0 col3" >1</td>
                        <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row0_col4" class="data row0 col4" >1</td>
            </tr>
            <tr>
                                <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row1_col0" class="data row1 col0" >0</td>
                        <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row1_col1" class="data row1 col1" >1</td>
                        <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row1_col2" class="data row1 col2" >0</td>
                        <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row1_col3" class="data row1 col3" >0</td>
                        <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row1_col4" class="data row1 col4" >0</td>
            </tr>
            <tr>
                                <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row2_col0" class="data row2 col0" >1</td>
                        <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row2_col1" class="data row2 col1" >0</td>
                        <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row2_col2" class="data row2 col2" >0</td>
                        <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row2_col3" class="data row2 col3" >0</td>
                        <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row2_col4" class="data row2 col4" >0</td>
            </tr>
            <tr>
                                <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row3_col0" class="data row3 col0" >1</td>
                        <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row3_col1" class="data row3 col1" >1</td>
                        <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row3_col2" class="data row3 col2" >1</td>
                        <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row3_col3" class="data row3 col3" >0</td>
                        <td id="T_1713c8d2_54df_11eb_8e7f_40a3cc65d4e3row3_col4" class="data row3 col4" >1</td>
            </tr>
    </tbody></table>



# Neural network multi-class classification
Multiclass classification in neural network is an extension of the on vs all method. Let's say that we want to build an image processing algorithm that can distinguish between four class of vehicles. We will build a neural network with 4 output units, each of which will model one of the output classes $C$

$$
h\Theta(x) = \begin{bmatrix}
P(y_1 \mid x, \Theta) \\
P(y_2 \mid x, \Theta) \\
P(y_3 \mid x, \Theta)\\
P(y_4 \mid x, \Theta)
\end{bmatrix}
$$


![png](ML-14-NeuralNetworksApplications_files/ML-14-NeuralNetworksApplications_25_0.png)


So that $h^{(i)}_\Theta(x)$ can be one of the following

$$h^{(i)}_\Theta(x) \approx \begin{bmatrix}1\\0\\0\\0\end{bmatrix} \;, \;  \begin{bmatrix}0\\1\\0\\0\end{bmatrix} \;, \;  \begin{bmatrix}0\\0\\1\\0\end{bmatrix}
\;, \;  \begin{bmatrix}0\\0\\0\\1\end{bmatrix}$$

