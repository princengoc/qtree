import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def plotComparison(G_est, G_true, mapping = 'identity', nodesize=10):
    """
    Plots the G_est with G_true as the ground truth graph. 
    Correct edges are colored blue, missing
    edges are gray, wrong edges (j,i) with i downstream j brown and red otherwise 
        
    Input Parameters
    ----------
    G_est : networkx DAG. 
    G_true : nextworkx DAG
    mapping: 
      If it is 'identity', then G_est and G_true have the same node labels
      Otherwise, this is a dictionary of the form {(node in estimated G, list of corresponding nodes in G_true)}
    nodesize : size of the nodes in the plot (optional, default=10)
    
    Examples
    --------
    Create a tree G: 1 -> 2 -> 3 -> 4 and plot it    
    >>> adj_list=adj_list= [['1', '2'], ['2', '3'], ['3', '4']]
    >>> G=nx.DiGraph()
    >>> G.add_edges_from(adj_list)
    >>> plotComparison(G,G)
    
    Contract G to create a graph H and plot H vs G
    
    >>> mapping = {'1,2': ('1', '2'), '4': '4'}
    >>> H = contract(G,mapping)
    >>> plotComparison(H,G,mapping)

    """            
    if mapping == 'identity':
      assert set(G_est.nodes) == set(G_true.nodes)
      G = G_true
    else:
      G = contract(G_true,mapping)

    #make sure that node names are str
    G = nx.relabel_nodes(G, mapping = dict([[i,str(i)] for i in G.nodes()]), copy=True)
    [pos_nodes, sink]=calculatePositions(G)
    colors=setEdgeColors(G_est, G)
    
    x_val=[pos_nodes[x][0] for x in pos_nodes]
    y_val=[pos_nodes[x][1] for x in pos_nodes]
    
    nodecolor='lightskyblue'
    node_size=min(6/(max(x_val)-min(x_val)+1),4/(max(y_val)-min(y_val)+1))

    plt.close()
    ax = plt.gca()
    
    bbox_oval=dict(boxstyle="round4", fc=(0,0,0,0), ec=(0,0,0,0), pad=0.9)
    bbox_round=dict(boxstyle="circle", fc=(0,0,0,0), ec=(0,0,0,0), pad=0.9)
    
    #Plot Nodes
    max_label_size=0
    
    for node in G.nodes:
        if len(node.split(','))==1:
            if len(node)>max_label_size:
                max_label_size=len(node)
    
    #Plot Edges   
    
    for edge in colors:
        color=colors[edge]
        if len(edge[1].split(','))==1:
            box=bbox_round
            text=' '*max_label_size
        else:
            box=bbox_oval
            text=edge[1]
        if color=='r' or color=='sienna':
            arc=str(-0.4-np.random.normal(scale=0.1))
            ann = ax.annotate(text, xy=pos_nodes[edge[0]],xytext=pos_nodes[edge[1]],
                  fontweight="bold", size=nodesize, va="center", 
                  ha="center", bbox=box, arrowprops=dict(arrowstyle="<|-,head_length=0.5,head_width=0.25", 
                  lw=nodesize/10, connectionstyle="arc3,rad="+ arc, fc=color, ec=color)
                  )
        else: 
            ann = ax.annotate(text,
                  xy=pos_nodes[edge[0]],xytext=pos_nodes[edge[1]], fontweight="bold", size=nodesize,
                  va="center", ha="center", bbox=box, 
                  arrowprops=dict(arrowstyle="<|-,head_length=0.5,head_width=0.25",lw=nodesize/10, fc=color, ec=color)    
                  )
            
    
    for node in G.nodes:
        if node==sink[0]:
            curr_color='yellow'
        else:
            curr_color=nodecolor
        
        if len(node.split(','))==1:
            box_style="circle"
            ann = ax.annotate(' '*max_label_size,
            xy=pos_nodes[node], xytext=pos_nodes[node],fontweight="bold", size=nodesize,
            va="center", ha="center", 
            bbox=dict(boxstyle=box_style, fc=curr_color, ec=curr_color, pad=0.9)
            )
            
            
            ann = ax.annotate(node,
            xy=pos_nodes[node], xytext=pos_nodes[node],fontweight="bold", size=nodesize,
            va="center", ha="center", 
            bbox=dict(boxstyle=box_style, fc=(0,0,0,0), ec=(0,0,0,0))
            )          
            
        else:
            box_style="round4"  
            ann = ax.annotate(node,
            xy=pos_nodes[node], xytext=pos_nodes[node],fontweight="bold", size=nodesize,
            va="center", ha="center", 
            bbox=dict(boxstyle=box_style, fc=curr_color, ec=curr_color, pad=0.9)
            )
              
    plt.xlim((min(x_val)-10, max(x_val)+10))    
    plt.ylim((min(y_val)-10, max(y_val)+10))
    plt.axis('off')


def getRiver_fromAdj(fname):
  """Returns a networkx DiGraph built with the given adjacency list. 
  Node labels are converted to str by default. 
  
  'ColoradoAdjacencyList.txt'
  """
  adj_list=np.loadtxt(fname,dtype = 'str')
  G = nx.DiGraph()
  G.add_edges_from(adj_list)
  return G

def contract(G, mapping):
    """
    Returns the contraction H of G onto the given a dictionary of nodes. 
    Contraction edges are defined based on the reachability criterion. 
    
    Input Parameters
    ----------
    G : a networkx DAG, whose node labels are strings.

    mapping:  
      A dictionary with the old labels as keys and A LIST OF NODES of G as values. 
      
    Example
    --------
    G = 1 -> 2 ->  3 -> 4
    contract to '1,2', '4' gives the graph
    H = '1,2' -> 4
    
    >>> adj_list=adj_list= [['1', '2'], ['2', '3'], ['3', '4']]
    >>> G=nx.DiGraph()
    >>> G.add_edges_from(adj_list)
    >>> mapping = {'1,2': ('1', '2'), '4': '4'}
    >>> H = contract(G,mapping)
    """
    assert nx.is_directed_acyclic_graph(G)
    H = nx.DiGraph()
    H.add_nodes_from(mapping.keys())
    
    #compute the inverse mapping: from the nodes of G to the nodes of H
    mapping_inverse = dict()
    for key,value in mapping.items():
      if type(value) is list or type(value) is tuple:
        for i in value:
          mapping_inverse[i] = key
      else:
        mapping_inverse[value] = key

    for v in mapping.keys():
      cluster = mapping[v]
      if type(cluster) is not list and type(cluster) is not tuple:
        cluster = [cluster]
      for i in cluster:
          #children sorted by path length
          children_sorted = [k for k in nx.shortest_path_length(G,i) if k in mapping_inverse]
          if len(children_sorted) > 1:
            child = children_sorted[1] 
            H.add_edge(v, mapping_inverse[child]) #add the direct child, if any.
          #parents sorted by path length
          parents_sorted = [k for k in nx.shortest_path_length(G,source=None, target = i) if k in mapping_inverse]
          if len(parents_sorted) > 1:
            parent = parents_sorted[1] 
            H.add_edge(mapping_inverse[parent],v) #add the direct parent, if any.
    #remove self-loops
    H.remove_edges_from(nx.selfloop_edges(H))                
    return H


def calculatePositions(G):
    
    """
    Calculates the (x,y) coordinates for each node in G
    
    Input Parameters
    ----------
    G : networkx DiGraph, DAG with a unique sink
    """
    sink=[i for i in G.nodes() if len([j for j in G.successors(i)]) == 0 ]    
    assert len(sink)==1

    pos_nodes={}
    x=0
    
    s=np.copy(sink)
    while s:
        y=0
        max_cluster=0
        max_cluster=max([max_cluster]+[len(i.split(',')) for i in s])
        print(max_cluster)
        x+=5+(max_cluster-1)*3
        for k in s:
            for l in list(G.successors(k)):
                if l in pos_nodes and pos_nodes[l][1]<y:
                    y=pos_nodes[l][1]
            pos_nodes.update({k:(x,y)})
            y=y-10
        x+=5+(max_cluster-1)*3    
        s_t=[]
        for k in s:
            s_t=s_t+(list(G.predecessors(k)))
        s=list(dict.fromkeys(s_t))
       
    return [pos_nodes, sink]

def setEdgeColors(G_est, G):
    
    """
    Calculates the edge coloring - Edges that are in the (true) Graph G but not
    in G_est, are gray, common edges are blue, edges (i,j) that are in 
    G_est but not in G are brown if G has a path from i to j and red otherwise
    
    Input Parameters
    ----------
    G : networkx DiGraph 
    G_est : networkx DiGraph with identical node set as G

    """
    colors={}
        
    for edge in G_est.edges:
        if edge in G.edges:
            colors.update({edge:'b'})
        elif nx.has_path(G, edge[0], edge[1]):
            colors.update({edge:'sienna'})
        else:
            colors.update({edge:'r'})
            
    for edge in G.edges:
        if edge not in G_est.edges:  
            colors.update({edge:'gray'})
            
    return colors            
            
        
