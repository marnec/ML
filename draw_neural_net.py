import matplotlib.pyplot as plt
import numpy as np


def _add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = np.mean(xdata)
        
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, output=None, node_labels=None, edge_labels=None, 
    layer_labels=None, colors=None, edgecolors=None, bias_nodes=None, lwe=1, lwn=1):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''

    if output is not None:
        layer_sizes.append(1)

    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)

    layers = len(layer_sizes)

    # Nodes
    nodes = []
    radius = v_spacing/4
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2 + (top + bottom)/2
        x = n*h_spacing + left
        for m in range(layer_size):
            xy = (x, layer_top - m*v_spacing)
            if n + 1 == layers and output is not None:
                ax.text(*xy, f" {output}", fontsize=16, va='center', zorder=10)
            else:
                circle = plt.Circle(xy, radius, color='w', ec='k', zorder=4, lw=lwn)
                ax.add_artist(circle)
                nodes.append(circle)
        
        # Layer labels
        if layer_labels is not None:
            if layer_labels is True:
                if output is None or (output is not None and n+1 != layers):
                    ax.text(x, 0.05, f"Layer {n+1}", fontsize=12, va='center', ha='center', zorder=10)
            if isinstance(layer_labels, list):
                if n <= len(layer_labels)-1:
                    ax.text(x, 0.05, layer_labels[n], fontsize=12, va='center', ha='center', zorder=10)

    # Bias-Nodes
    if bias_nodes is not None:
        for n, bnode in zip(range(layers), bias_nodes):
            if bnode is not False:
                x_bias = (n)*h_spacing + left
                y_bias = top + 0.005
                circle = plt.Circle((x_bias, y_bias), v_spacing/4., color='w', ec='k', zorder=4, lw=lwn)
                ax.text(x_bias, y_bias, '+1', fontsize=12, zorder=10, va='center', ha='center')
                ax.add_artist(circle)   

    # Node labels
    if node_labels is not None and isinstance(node_labels, list):
        for node, label in zip(nodes, node_labels):
            ax.text(*node.center, label, fontsize=12, va='center', ha='center', zorder=10)

    # Node colors
    if colors is not None and isinstance(colors, list):
        for node, c in zip(nodes, colors):
            node.set_edgecolor(c)

    # Edges
    edges = []
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k', lw=lwe)
                edges.append(line)
                ax.add_artist(line)

                # Edge labels
                if edge_labels is not None:
                    elabel = ""
                    if isinstance(edge_labels, list):
                        if len(edges) <= len(edge_labels):
                            elabel = edge_labels[len(edges)-1]
                    elif isinstance(edge_labels, np.array):
                        elabel = str(round(edge_labels[n][m, o],4))

                    xm = (n*h_spacing + left)
                    xo = ((n + 1)*h_spacing + left)
                    ym = (layer_top_a - m*v_spacing)
                    yo = (layer_top_b - o*v_spacing)
                    rot_mo_rad = np.arctan((yo-ym)/(xo-xm))
                    rot_mo_deg = rot_mo_rad*180./np.pi
                    xm1 = xm + (v_spacing/2+0.05)*np.cos(rot_mo_rad)
                    if n == 0:
                        if yo > ym:
                            ym1 = ym + (v_spacing/2.+0.12)*np.sin(rot_mo_rad)
                        else:
                            ym1 = ym + (v_spacing/2+0.05)*np.sin(rot_mo_rad)
                    else:
                        if yo > ym:
                            ym1 = ym + (v_spacing/2+0.12)*np.sin(rot_mo_rad)
                        else:
                            ym1 = ym + (v_spacing/2+0.04)*np.sin(rot_mo_rad)
                    ax.text(xm1, ym1, elabel, rotation=rot_mo_deg, fontsize=10)

    # Edges between bias and nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if bias_nodes is not None and bias_nodes[n] is True:
            layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
            layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
            x_bias = n*h_spacing + left
            y_bias = top + 0.005 
            for o in range(layer_size_b):
                line = plt.Line2D([x_bias, (n + 1)*h_spacing + left],
                              [y_bias, layer_top_b - o*v_spacing], c='k', lw=lwe)
                edges.append(line)
                ax.add_artist(line)
                if edge_labels is not None:
                    elabel=""
                    if isinstance(edge_labels, list):
                        if len(edges) <= len(edge_labels):
                            elabel = edge_labels[len(edges)-1]
                    # elif isinstance(edge_labels, np.array):
                    #     elabel = str(round(edge_labels[n][m, o],4))
                    xo = ((n + 1)*h_spacing + left)
                    yo = (layer_top_b - o*v_spacing)
                    rot_bo_rad = np.arctan((yo-y_bias)/(xo-x_bias))
                    rot_bo_deg = rot_bo_rad*180./np.pi
                    xo2 = xo - (v_spacing+0.01)*np.cos(rot_bo_rad)
                    yo2 = yo - (v_spacing+0.01)*np.sin(rot_bo_rad)
                    xo1 = xo2 -0.05 *np.cos(rot_bo_rad)
                    yo1 = yo2 -0.05 *np.sin(rot_bo_rad)
                    plt.text( xo1, yo1, elabel, rotation=rot_bo_deg, fontsize=10)  

    if edgecolors is not None:
        for color, edge in zip(edgecolors, edges):
            edge.set_color(color)
                
    
    ax.axis('off')


def draw_annotated_neural_net(ax, left, right, bottom, top, layer_sizes, coefs_=None, intercepts_=None, n_iter_=None, loss_=None):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # Input-Arrows
    layer_top_0 = v_spacing*(layer_sizes[0] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[0]):
        plt.arrow(left-0.18, layer_top_0 - m*v_spacing, 0.12, 0,  lw =1, head_width=0.01, head_length=0.02)
    
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/8.,
                                color='w', ec='k', zorder=4)
            if n == 0:
                plt.text(left-0.125, layer_top - m*v_spacing, r'$X_{'+str(m+1)+'}$', fontsize=15)
            elif (n_layers == 3) & (n == 1):
                plt.text(n*h_spacing + left+0.00, layer_top - m*v_spacing+ (v_spacing/8.+0.01*v_spacing), r'$H_{'+str(m+1)+'}$', fontsize=15)
            elif n == n_layers - 1:
                plt.text(n*h_spacing + left+0.10, layer_top - m*v_spacing, r'$y_{'+str(m+1)+'}$', fontsize=15)
            ax.add_artist(circle)
    
    # Bias-Nodes
    for n, layer_size in enumerate(layer_sizes):
        if n < n_layers -1:
            x_bias = (n+0.5)*h_spacing + left
            y_bias = top + 0.005
            circle = plt.Circle((x_bias, y_bias), v_spacing/8., color='w', ec='k', zorder=4)
            plt.text(x_bias-(v_spacing/8.+0.10*v_spacing+0.01), y_bias, r'$1$', fontsize=15)
            ax.add_artist(circle)   
    
    # Edges
    # Edges between nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)

                if coefs_ is not None:
                    xm = (n*h_spacing + left)
                    xo = ((n + 1)*h_spacing + left)
                    ym = (layer_top_a - m*v_spacing)
                    yo = (layer_top_b - o*v_spacing)
                    rot_mo_rad = np.arctan((yo-ym)/(xo-xm))
                    rot_mo_deg = rot_mo_rad*180./np.pi
                    xm1 = xm + (v_spacing/8.+0.05)*np.cos(rot_mo_rad)
                    if n == 0:
                        if yo > ym:
                            ym1 = ym + (v_spacing/8.+0.12)*np.sin(rot_mo_rad)
                        else:
                            ym1 = ym + (v_spacing/8.+0.05)*np.sin(rot_mo_rad)
                    else:
                        if yo > ym:
                            ym1 = ym + (v_spacing/8.+0.12)*np.sin(rot_mo_rad)
                        else:
                            ym1 = ym + (v_spacing/8.+0.04)*np.sin(rot_mo_rad)
                    plt.text( xm1, ym1,str(round(coefs_[n][m, o],4)), rotation=rot_mo_deg, fontsize=10)
    
    # Edges between bias and nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if n < n_layers-1:
            layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
            layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        x_bias = (n+0.5)*h_spacing + left
        y_bias = top + 0.005 
        for o in range(layer_size_b):
            line = plt.Line2D([x_bias, (n + 1)*h_spacing + left],
                          [y_bias, layer_top_b - o*v_spacing], c='k')
            ax.add_artist(line)
            if intercepts_ is not None:
                xo = ((n + 1)*h_spacing + left)
                yo = (layer_top_b - o*v_spacing)
                rot_bo_rad = np.arctan((yo-y_bias)/(xo-x_bias))
                rot_bo_deg = rot_bo_rad*180./np.pi
                xo2 = xo - (v_spacing/8.+0.01)*np.cos(rot_bo_rad)
                yo2 = yo - (v_spacing/8.+0.01)*np.sin(rot_bo_rad)
                xo1 = xo2 -0.05 *np.cos(rot_bo_rad)
                yo1 = yo2 -0.05 *np.sin(rot_bo_rad)
                plt.text( xo1, yo1, str(round(intercepts_[n][o],4)), rotation=rot_bo_deg, fontsize=10)    
                
    # Output-Arrows
    layer_top_0 = v_spacing*(layer_sizes[-1] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[-1]):
        plt.arrow(right+0.015, layer_top_0 - m*v_spacing, 0.16*h_spacing, 0,  lw =1, head_width=0.01, head_length=0.02)
    
    if n_iter_ is not None and loss_ is not None:
        # Record the n_iter_ and loss
        plt.text(left + (right-left)/3., bottom - 0.005*v_spacing, \
                 'Steps:'+str(n_iter_)+'    Loss: ' + str(round(loss_, 6)), fontsize = 15)

    ax.set_aspect((top-bottom)/(right-left))
    ax.axis('off')
