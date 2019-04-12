#running in python 2.7
#extra lib numpy and graphviz

import numpy
import random
from collections import defaultdict 
from numpy import sum as npsum
from numpy import finfo, log, sqrt
from numpy.linalg import eigvalsh
from graphviz import Digraph
import warnings
warnings.filterwarnings("error")

epsilon = sqrt(finfo(float).eps)

def plogdet(K):
    r"""Log of the pseudo-determinant.

    It assumes that ``K`` is a positive semi-definite matrix.

    Args:
        K (array_like): matrix.

    Returns:
        float: log of the pseudo-determinant.
    """
    egvals = eigvalsh(K)
    return npsum(log(egvals[egvals > epsilon]))


class DenpendenceTree:
    def __init__(self,children,value,prob0,prob1,parent):
        self.children = children
        self.value = value
        self.prob0 = prob0
        self.prob1 = prob1
        self.parent = parent

    def __str__(self, level=0):
        ret = "\t"*level+repr(self.value)+"\t"+"prob0:"+repr(self.prob0)+"\t"+"prob1:"+repr(self.prob1)+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

class DecisionTree:
    def __init__(self,value,children,l1,l2,parent):
        self.value = value
        self.children = children
        self.l1 = l1
        self.l2 = l2
        self.parent = parent

    def __str__(self, level=0):
        ret = "\t"*level+repr(self.value)+"\n"
        for k, v in self.children.items():
            ret += str(k)+"-"+v.__str__(level+1)
        return ret

def makeprob():
    p = int(100*random.random())/100.0
    return p if p != 0.0 else 0.1
    
def expand(nodes,alist):
    layer = int(random.random()*len(alist))
    layer = layer if layer != 0 else 1
    newnodes = []
    for i in range(layer):
        newnodes.append(DenpendenceTree([],alist.pop(int(random.random()*len(alist))),makeprob(),makeprob(),None))
    for i in range(len(nodes)):
        if i == len(nodes)-1:
            nodes[i].children = newnodes
            for x in newnodes:
                x.parent = nodes[i]
        else:
            num = int(random.random()*len(newnodes))
            nodes[i].children = newnodes[0:num]
            for x in newnodes[0:num]:
                x.parent = nodes[i]
            newnodes = newnodes[num:]
    return nodes

def modelfakedata():
    fake_data = []
    for x in range(2000):
        temp = []
        for y in range(10):
            temp.append(int(2*random.random()))
        fake_data.append(temp)
    return fake_data

def findmean(alist):
    means = []
    for x in range(10):
        means.append(numpy.mean([i[x] for i in alist]))
    return means

def findvarnaive(alist):
    var = []
    for x in range(10):
        var.append([0]*x+[numpy.var([i[x] for i in alist])]+[0]*(9-x))
    return var

def findvaroptimal(alist):
    var = []
    for x in range(10):
        var.append(numpy.var([i[x] for i in alist]))
    M = []
    for i in range(10):
        temp = []
        for y in range(10):
            temp.append(var[y]*var[i])
        M.append(temp)
    return M

def D(x,M,E):
    XmM = numpy.subtract(x,M)
    first = numpy.dot(numpy.transpose(XmM),numpy.linalg.inv(E))
    return numpy.dot(first,XmM)

def optimalD(x,M,E):
    XmM = numpy.subtract(x,M)
    try:
        E = numpy.linalg.inv(E)
    except:
        E = numpy.linalg.pinv(E)
    first = numpy.dot(numpy.transpose(XmM),E)
    return numpy.dot(first,XmM)

def naivebayes(set1,set2,alist):
    mean1 = findmean(set1)
    mean2 = findmean(set2)
    var1 = findvarnaive(set1)
    var2 = findvarnaive(set2)
    return numpy.log(numpy.linalg.det(var2)) - numpy.log(numpy.linalg.det(var1)) + D(alist,mean2,var2) - D(alist,mean1,var1)

def optimalbayes(set1,set2,alist):
    mean1 = findmean(set1)
    mean2 = findmean(set2)

    var1 = findvaroptimal(set1)
    var2 = findvaroptimal(set2)
    try:
        lnA = numpy.log(numpy.linalg.det(var2))
    except:
        lnA = plogdet(var2)
    try:
        lnB = numpy.log(numpy.linalg.det(var1))
    except: 
        lnB = plogdet(var2)
    return lnA - lnB + optimalD(alist,mean2,var2) - optimalD(alist,mean1,var1)

def entropy(alist):
    sum1 = sum(alist)
    en = 0
    for x in alist:
        x = x*1.0
        if sum1 != 0 and x/sum1 != 0:
            en += -(x/sum1)*numpy.log2(x/sum1)
    return en

def makedecisiontreelayer(alist,node,attribute):
    childlist = []
    for i in attribute:
        index,newattribute,l1,l2 = gain1(node.l1,node.l2,alist,[node.value,i])
        if len(l1) == 0:
            node.children[i] = DecisionTree("c2",{},l1,l2,node)
        elif len(l2) == 0:
            node.children[i] = DecisionTree("c1",{},l1,l2,node)
        if index == -1:
            if len(l1)>len(l2):
                node.children[i] = DecisionTree("c1",{},l1,l2,node)
            else:
                node.children[i] = DecisionTree("c2",{},l1,l2,node)
        else:
            alist[index] = -1
            node.children[i] = DecisionTree(index,{},l1,l2,node)
            childlist.append(node.children[i])
    #print [i.value for i in childlist]
    return childlist


def combinlist(list1,list2):
    newlist = list1[:]
    for i in list2:
        newlist.append(i)
    return newlist

def combinlists(list):
    newlist = []
    for i in list:
        for x in i:
            newlist.append(x)
    return newlist

def gain1(class1,class2,alist,choose = [-1,0]):
    list1 = class1[:]
    list2 = class2[:]
    if choose[0] != -1:
        list1 = [i for i in list1 if i[choose[0]] == choose[1]]
        list2 = [i for i in list2 if i[choose[0]] == choose[1]]
    sum1 = len(list1)+len(list2)
    attrs = 0
    S = entropy([len(list1),len(list2)])
    attrlist = []
    for x in range(len(alist)):
        if alist[x] == -1:
            attrlist.append([-1,[-1,-1]])
        else:
            b = []
            [b.append(i[x]) for i in combinlist(list1,list2) if i[x] not in b]
            sumattr = 0
            for y in b:
                attributelist11 = len([i[x] for i in list1 if i[x]==y])
                attributelist21 = len([i[x] for i in list2 if i[x]==y])
                sumattr += numpy.abs((attributelist11+attributelist21)*1.0/sum1)*entropy([attributelist11,attributelist21])
            attrlist.append([S - sumattr,b])
    value = max([i[0] for i in attrlist])
    ind = [i[0] for i in attrlist].index(value) if value != -1 else -1
    return ind,attrlist[ind][1],list1,list2

def makeaDT(x):
    alist = range(x)
    root = DenpendenceTree([],alist.pop(int(random.random()*len(alist))),makeprob(),None,None)
    currentnode = [root]
    while(len(alist)):
        expand(currentnode,alist)
        newcurrent = []
        for i in currentnode:
            newcurrent.extend(i.children)  
        currentnode =newcurrent
    return root

def findattribute(alist,x):
    b = []
    [b.append(i[x]) for i in alist if i[x] not in b]
    return b

def findweight(allL,a,b):
    atx = findattribute(allL,a)
    aty = findattribute(allL,b)
    weight = 0
    for x in atx:
        for y in atx:
            prx = 0
            pry = 0
            prxandy = 0
            for i in allL:
                if i[a] == x:
                    prx += 1
                if i[b] == y:
                    pry += 1
                if i[a] == x and i[b] == y:
                    prxandy+=1
            #print prx,pry,prxandy
            prx = prx*1.0/len(allL)
            pry = pry*1.0/len(allL)
            prxandy = prxandy*1.0/len(allL)
           # print prxandy
            if prx*pry!=0 and prxandy!=0:
                weight+=prxandy*numpy.log(prxandy/(prx*pry))
    return weight

def findgraph(allL,l = [0,1,2,3,4,5,6,7,8,9]):
    edge = []
    for x in l:
        for y in l:
            if x!=y:
                edge.append([x,y,findweight(allL,x,y)])
    return edge

def findMST(l,a=None):
    g = Graph(len(l[0]))
    if a==None:
        edge = findgraph(l)
    else:
        edge = findgraph(l,a)
    for i in edge:
        g.addEdge(i[0],i[1],i[2])
    return g.KruskalMST()

def findDependeceT(allL,l=None):
    if l==None:
        mst = [i[:2] for i in findMST(allL)]
    else:
        mst = [i[:2] for i in findMST(allL,l)]
    root = DenpendenceTree([],mst[0][0],0,0,None)
    expand = [root]
    seen = []
    while len(expand)>0:
        current = expand.pop(0)
        seen.append(current.value)
        for i in mst:
            if current.value == i[0]:
                if not i[1] in seen:
                    node = DenpendenceTree([],i[1],0,0,None)
                    expand.append(node)
                    current.children.append(node)
            elif current.value == i[1]:
                if not i[0] in seen:
                    node = DenpendenceTree([],i[0],0,0,None)
                    expand.append(node)
                    current.children.append(node)

    prb = 0
    for i in allL:
        if i[root.value] == 0:
            prb+=1
    
    root.prob0 = prb*1.0/len(allL)
    findprob(root,allL)

    f = [root]
    while f!=[]:
        current = f.pop(0)
        for i in current.children:
            i.parent = current
        f.extend(current.children)
    return root
             
def findprob(root,allL):
    if len(root.children)==0:
        return
    else:
        prob0 = 0
        prob1 = 0
        for i in root.children:
            for x in allL:
                if x[root.value] == 0 and x[i.value] == 0:
                    prob0 += 1
                elif x[root.value] == 1 and x[i.value] == 0:
                    prob1 += 1
            i.prob0 = prob0*1.0/len(allL)
            i.prob1 = prob1*1.0/len(allL)
            findprob(i,allL)
    

def makeData(tree):
    newdata = [-1]*10
    alist = [tree]
    while len(alist)>0:
        currentnode = alist.pop(0)
        alist.extend(currentnode.children)
        value = random.random()
        if currentnode.parent != None:
            if newdata[currentnode.parent.value] == 0:
                if value <= currentnode.prob0:
                    newdata[currentnode.value] = 0
                else:
                    newdata[currentnode.value] = 1
            else:
                if value <= currentnode.prob1:
                    newdata[currentnode.value] = 0
                else:
                    newdata[currentnode.value] = 1
        else:
            if value <= currentnode.prob0:
                newdata[currentnode.value] = 0
            else:
                newdata[currentnode.value] = 1
    return newdata

def makerelation(node,children):
    node.children = children
    for i in children:
        i.parent = node

def testDT1():
    node0 = DenpendenceTree([],0,0.402677629947,0.0617801891226,None)
    node1 = DenpendenceTree([],1,0.772556389465,0.772992605709,None)
    node2 = DenpendenceTree([],2,0.644046259213,0.785863016096,None)
    node3 = DenpendenceTree([],3,0.756899686359,0.0540867117931,None)
    node4 = DenpendenceTree([],4,0.4936665592,0.829181546968,None)
    node5 = DenpendenceTree([],5,0.626963057627,0.821448049196,None)
    node6 = DenpendenceTree([],6,0.425209535123,0.389736256434,None)
    node7 = DenpendenceTree([],7,0.753422537982,0.0141071748507,None)
    node8 = DenpendenceTree([],8,0.304157348143,0.645107616455,None)
    node9 = DenpendenceTree([],9,0.456501754264,0.420513461728,None)

    makerelation(node8,[node6,node3])
    makerelation(node6,[node2])
    makerelation(node3,[node0,node9])
    makerelation(node2,[node7])
    makerelation(node0,[node5,node4,node1])
    return node8

def testDT2():
    node0 = DenpendenceTree([],0, 0.31 , 0.89 ,None)
    node1 = DenpendenceTree([],1, 0.05 , 0.26 ,None)
    node2 = DenpendenceTree([],2, 0.99 , 0.32 ,None)
    node3 = DenpendenceTree([],3, 0.71 , 0.09 ,None)
    node4 = DenpendenceTree([],4, 0.9 , 0.53 ,None)
    node5 = DenpendenceTree([],5, 0.45 , 0.2 ,None)
    node6 = DenpendenceTree([],6, 0.26 , 0.51 ,None)
    node7 = DenpendenceTree([],7, 0.83 , 0.36 ,None)
    node8 = DenpendenceTree([],8, 0.44 , 0.07 ,None)
    node9 = DenpendenceTree([],9, 0.44 , 0.13 ,None)

    makerelation(node8,[node6,node3])
    makerelation(node6,[node2])
    makerelation(node3,[node0,node9])
    makerelation(node2,[node7])
    makerelation(node0,[node5,node4,node1])
    return node8

def testDT3():
    node0 = DenpendenceTree([],0, 0.59 , 0.07 ,None)
    node1 = DenpendenceTree([],1, 0.28 , 0.14 ,None)
    node2 = DenpendenceTree([],2, 0.22 , 0.38 ,None)
    node3 = DenpendenceTree([],3, 0.24 , 0.74 ,None)
    node4 = DenpendenceTree([],4, 0.96 , 0.64 ,None)
    node5 = DenpendenceTree([],5, 0.32 , 0.59 ,None)
    node6 = DenpendenceTree([],6, 0.18 , 0.36 ,None)
    node7 = DenpendenceTree([],7, 0.71 , 0.42 ,None)
    node8 = DenpendenceTree([],8, 0.65 , 0.07 ,None)
    node9 = DenpendenceTree([],9, 0.51 , 0.64 ,None)

    makerelation(node8,[node6,node3])
    makerelation(node6,[node2])
    makerelation(node3,[node0,node9])
    makerelation(node2,[node7])
    makerelation(node0,[node5,node4,node1])
    return node8

def testDT4():
    node0 = DenpendenceTree([],0, 0.96 , 0.75 ,None)
    node1 = DenpendenceTree([],1, 0.88 , 0.64 ,None)
    node2 = DenpendenceTree([],2, 0.81 , 0.93 ,None)
    node3 = DenpendenceTree([],3, 0.97 , 0.4 ,None)
    node4 = DenpendenceTree([],4, 0.52 , 0.74 ,None)
    node5 = DenpendenceTree([],5, 0.6 , 0.55 ,None)
    node6 = DenpendenceTree([],6, 0.51 , 0.45 ,None)
    node7 = DenpendenceTree([],7, 0.34 , 0.59 ,None)
    node8 = DenpendenceTree([],8, 0.89 , 0.29 ,None)
    node9 = DenpendenceTree([],9, 0.15 , 0.49 ,None)

    makerelation(node8,[node6,node3])
    makerelation(node6,[node2])
    makerelation(node3,[node0,node9])
    makerelation(node2,[node7])
    makerelation(node0,[node5,node4,node1])
    return node8

def makedecisiontree(class1,class2,alist):
    index,attribute,l1,l2, = gain1(class1,class2,alist)
    alist[index] = -1
    root = DecisionTree(index,{},l1,l2,None)
    childlist = [root]
    while(len(childlist)>0):
        node = childlist.pop(0)
        if node.value in range(10):
            childlist.extend(makedecisiontreelayer(alist,node,attribute))
    return root

def Crossvalidation5DT(alist,blist,list1):
    apart = len(alist)/5
    bpart = len(blist)/5
    adata = [alist[apart*i:apart*(i+1)] for i in range(5)]
    bdata = [blist[bpart*i:bpart*(i+1)] for i in range(5)]
    all = 0
    for i in range(5):
        atestdata = adata[i]
        btestdata = bdata[i]
        l1 = list1[:]
        atranning = adata[:].pop(i)
        btranning = bdata[:].pop(i)
        root = makedecisiontree(atranning,btranning,l1)
        drawDecideTree(root)
        value = 0
        for i in atestdata:
            if decide(i,root) == "c1":
                value+=1
        all += value*1.0/apart
    print all/5

def bayesindependent(alist,check):
    l1 = alist[0]
    result = 1
    for i in range(1,len(alist)):
        result = result if independentBY(l1,alist[i],check)>=0 else i+1
    print "it is in class",result

def bayesdependent(alist,check,roots):
    l1 = alist[0]
    r1 = roots[0]
    result = 1
    for i in range(1,len(alist)):
        if dependentBY(l1,alist[i],check,r1,roots[i])<0:
            result = i+1
            r1 = roots[i]
    print "it is in class",result


def Crossvalidation5OPTBayes(alist,blist):
    apart = len(alist)/5
    bpart = len(blist)/5
    adata = [alist[apart*i:apart*(i+1)] for i in range(5)]
    bdata = [blist[bpart*i:bpart*(i+1)] for i in range(5)]
    all = 0
    for i in range(5):
        atestdata = adata[i]
        btestdata = bdata[i]
        atranning = adata[:].pop(i)
        btranning = bdata[:].pop(i)
        value = 0
        for i in atestdata:

            v = naivebayes(atranning,btranning,i)
            if v>=0:
                value+=1
        all += value*1.0/apart
    print all/5


def Crossvalidation5Bayes(alist,blist,tree1=None,tree2 = None):
    apart = len(alist)/5
    bpart = len(blist)/5
    adata = [alist[apart*i:apart*(i+1)] for i in range(5)]
    bdata = [blist[bpart*i:bpart*(i+1)] for i in range(5)]
    all = 0
    for i in range(5):
        atestdata = adata[i]
        btestdata = bdata[i]
        atranning = adata[:].pop(i)
        btranning = bdata[:].pop(i)
        value = 0
        for i in atestdata:
            if tree1 == None:
                v = independentBY(atranning,btranning,i)
            else:
                v = dependentBY(atranning,btranning,i,tree1,tree2)
            if v>=0:
                value+=1
        all += value*1.0/apart
    print all/5

def decide(l,root):
    node = root
    while bool(node.children):
        node = node.children[l[root.value]]
    return node.value

def moredecisiontree(alist,attrl,check):
    l1 = alist[0]
    result = 1
    for i in range(1,len(alist)):
        root = makedecisiontree(l1,alist[i],[0,1,2,3,4,5,6,7,8,9])
        drawDecideTree(root)
        print root
        if decide(check,root) == "c2":
            l1 = alist[i]
            result = i+1
    print "it's in class",result

def independentBY(l1,l2,check):
    value = 0
    prob1 = []
    prob2 = []
    for i in range(len(l1[0])):
        v1 = 0
        v2 = 0
        for x in l1:
            if x[i] == 1:
                v1+=1
        for y in l2:
            if y[i] == 1:
                v2+=1
        prob1.append(v1*1.0/len(l1))
        prob2.append(v2*1.0/len(l2))
        
    for i in range(len(l1[0])):
        if prob1[i]!=0 and prob2[i]!=0 and prob1[i]!=1 and prob2[i]!=1:
            value += check[i]*numpy.log(prob1[i]/prob2[i])+(1-check[i])*numpy.log((1-prob1[i])/(1-prob2[i]))
    value += numpy.log(len(l1)*1.0/len(l2))

    return value

def findNode(tree, value):
    L = [tree]
    while L!=[]:
        node = L.pop(0)
        if node.value == value:
            return node
        L.extend(node.children)
    return None

def dependentBY(l1,l2,check,tree1,tree2):
    value = 0
    for i in range(len(l1[0])):
        node1 = findNode(tree1,i)
        if node1.parent != None:
            prob1 = (1-node1.prob0) if check[node1.parent.value] == 0 else (1-node1.prob1)
        else:
            prob1 = 1- node1.prob0
        node2 = findNode(tree2,i)
        if node2.parent != None:
            prob2 = (1-node2.prob0) if check[node2.parent.value] == 0 else (1-node2.prob1)
        else:
            prob2 = 1- node2.prob0
        prob1 = abs(prob1)
        prob2 = abs(prob2)
        if prob1==0 or prob2==0:
            value += (1-check[i])*numpy.log((1-prob1)/(1-prob2))
        elif (1-prob1)==0 or (1-prob2)==0:
            value += check[i]*numpy.log(prob1/prob2)
        else:
            value += check[i]*numpy.log(prob1/prob2)+(1-check[i])*numpy.log((1-prob1)/(1-prob2))
    value += numpy.log(len(l1)*1.0/len(l2))
    return value

def writein(filename,l):
    with open(filename, 'w') as filehandle:  
        for i in l:
            filehandle.write('%s\n' % i)

class Graph: 
  
    def __init__(self,vertices): 
        self.V= vertices 
        self.graph = [] 

    def addEdge(self,u,v,w): 
        self.graph.append([u,v,w]) 
  
    def find(self, parent, i): 
        if parent[i] == i: 
            return i 
        return self.find(parent, parent[i]) 
  
    def union(self, parent, rank, x, y): 
        xroot = self.find(parent, x) 
        yroot = self.find(parent, y) 
  
        if rank[xroot] < rank[yroot]: 
            parent[xroot] = yroot 
        elif rank[xroot] > rank[yroot]: 
            parent[yroot] = xroot 
  
        else : 
            parent[yroot] = xroot 
            rank[xroot] += 1

    def KruskalMST(self): 
        result =[]
        i = 0 
        e = 0
        self.graph =  sorted(self.graph,key=lambda item: item[2])
        self.graph.reverse()

        parent = []
        rank = [] 
  
        for node in range(self.V): 
            parent.append(node) 
            rank.append(0) 

        while e < self.V -1 : 
  
            u,v,w =  self.graph[i] 
            i = i + 1
            x = self.find(parent, u) 
            y = self.find(parent ,v) 
  
            if x != y: 
                e = e + 1     
                result.append([u,v,w]) 
                self.union(parent, rank, x, y)         
        return result

def Crossvalidation5DT2(alist,blist,list1):
    apart = len(alist)/5
    bpart = len(blist)/5
    adata = [alist[apart*i:apart*(i+1)] for i in range(5)]
    bdata = [blist[bpart*i:bpart*(i+1)] for i in range(5)]
    all = 0
    for i in range(5):
        atestdata = adata[i]
        btestdata = bdata[i]
        l1 = list1[:]
        atranning = adata[:].pop(i)
        btranning = bdata[:].pop(i)
        root = makedecisiontree(atranning,btranning,l1)
        value = 0
        for i in atestdata:
            if decide(i,root) == "c1":
                value+=1
        all += value*1.0/apart
    print all/5

def readfile(filename,l):
    with open(filename, 'r') as filehandle:  
        for line in filehandle:
            currentPlace = line[1:len(line)-2]
            l.append(map(int,currentPlace.split(",")))

def readglassfile(filename,l):
    with open(filename, 'r') as filehandle:  
        for line in filehandle:
            l.append(map(float,line.split(",")))

def moreCVDT(alist,l):
    Crossvalidation5DT(alist[0],alist[1],l)
    Crossvalidation5DT(alist[1],alist[3],l)
    Crossvalidation5DT(alist[2],alist[3],l)
    Crossvalidation5DT(alist[3],alist[2],l)

def moreCVBY(alist,l2=None):
    if l2 ==None:
        Crossvalidation5Bayes(alist[0],alist[1])
        Crossvalidation5Bayes(alist[1],alist[3])
        Crossvalidation5Bayes(alist[2],alist[3])
        Crossvalidation5Bayes(alist[3],alist[2])
    else:
        Crossvalidation5Bayes(alist[0],alist[1],l2[0],l2[3])
        Crossvalidation5Bayes(alist[1],alist[3],l2[1],l2[3])
        Crossvalidation5Bayes(alist[2],alist[3],l2[2],l2[3])
        Crossvalidation5Bayes(alist[3],alist[2],l2[3],l2[2])

def moreOPTBY(alist):
    Crossvalidation5OPTBayes(alist[0],alist[1])
    Crossvalidation5OPTBayes(alist[1],alist[3])
    Crossvalidation5OPTBayes(alist[2],alist[3])
    Crossvalidation5OPTBayes(alist[3],alist[2])

def findrelationoftree(t1):
    info = {}
    q = [t1]
    while q!=[]:
        current = q.pop(0)
        temp = set()
        if current.parent!=None:
            temp.add(current.parent.value)
        for x in current.children:
            temp.add(x.value)
        info[current.value] = temp
        q.extend(current.children)
    return info

def comparetree(t1,t2):
    v1 = findrelationoftree(t1)
    v2 = findrelationoftree(t2)
    v = 0
    for i in range(10):
        set1 = v1[i]
        set2 = v2[i]
        if len(set1.union(set2)) != 0:
            v += len(set1.intersection(set2))*1.0/len(set1.union(set2))
        else:
            v+=1
    print v/10


def drawDenpendenceTree(tree):
    dot = Digraph(comment = 'denpendenceTree')
    d = [tree]
    dot.node(str(tree.value),str(tree.value)+"\n"+"P(X"+str(tree.value)+"=0):"+str(tree.prob0))
    while d!=[]:
        current = d.pop(0)
        for i in current.children:
            s0 = "P(X"+str(i.value)+"=0|X"+str(current.value)+"=0)"
            s1 = "P(X"+str(i.value)+"=0|X"+str(current.value)+"=1)"
            dot.node(str(i.value),str(i.value)+"\n"+s0+str(i.prob0)+"\n"+s1+str(i.prob1))
            dot.edge(str(current.value),str(i.value))
            d.append(i)
    dot.view()

def drawDecideTree(tree):
    dot = Digraph(comment = 'DecideTree')
    d = [tree]
    dot.node(str(tree.value),str(tree.value))
    while d!=[]:
        current = d.pop(0)
        for x,y in current.children.items():
            dot.node(str(y.value),str(y.value))
            dot.edge(str(current.value),str(y.value),str(x))
            d.append(y)
    dot.view()

TH = [1.5184,13.4079,2.6845,1.4449,72.6509,0.4971,8.9570,0.1750,0.0570]
def modGlassData(alist):
    modl = []
    for x in alist:
        temp = []
        for i in range(1,len(alist[0])-1):
            if x[i] < TH[i-1]:
                temp.append(0)
            else:
                temp.append(1)
        modl.append(temp)
    return modl

class1 = []
class2 = []
class3 = []
class4 = []
root1 = testDT1()
root2 = testDT2()
root3 = testDT3()
root4 = testDT4()

readfile("class1",class1)
readfile("class2",class2)
readfile("class3",class3)
readfile("class4",class4)

'''for i in range(2000):
    class1.append(makeData(root1))
    class2.append(makeData(root2))
    class3.append(makeData(root3))
    class4.append(makeData(root4))'''

alist = [class1,class2,class3,class4]
treelist = [root1,root2,root3,root4]

r1 = findDependeceT(class1)
r2 = findDependeceT(class2)
r3 = findDependeceT(class3)
r4 = findDependeceT(class4)
testtree = [r1,r2,r3,r4]

#1.
#moreCVDT(alist,[0,1,2,3,4,5,6,7,8,9]) #5-fold cross-validation for decision tree
#moreCVBY(alist) #5-fold cross-validation for bayes by binary distribution
#moreOPTBY(alist) #5-fold cross-validation for bayes by Gaussian distribution
#moreCVBY(alist,testtree) #5-fold cross-validation for bayes by binary distribution using dependence tree
#moreCVBY(alist,treelist) #5-fold cross-validation for bayes by binary distribution using orignal dependence tree

#2.
print "similarity of class 1:",comparetree(testtree[0],root1) #find the similarity of two dependence tree 
print "similarity of class 2:",comparetree(testtree[1],root2)
print "similarity of class 3:",comparetree(testtree[2],root3)
print "similarity of class 4:",comparetree(testtree[3],root4)
#drawDenpendenceTree(testtree[0]) # draw dependence tree 
#drawDenpendenceTree(testtree[1])
#drawDenpendenceTree(testtree[2])
#drawDenpendenceTree(testtree[3])

#3.
#bayesindependent([class1,class2,class3,class4],[1,0,0,1,0,1,0,1,0,1]) #question 1.2.3

#4.
#bayesdependent([class1,class2,class3,class4],[1,0,1,1,0,1,0,1,0,1],[root1,root2,root3,root4]) #qustion 1.2.4

#5.
#moredecisiontree([class1,class2,class3,class4],[0,1,2,3,4,5,6,7,8,9],class4[0]) #question 1.2.5

root = makedecisiontree(class1,class4,[0,1,2,3,4,5,6,7,8,9])
drawDecideTree(root)


#1.3
l = []
readglassfile("glass",l)
nonwindow = []
window = []
for i in l:
    if i[-1] in [5.0,6.0,7.0]:
        nonwindow.append(i)
    else:
        window.append(i)

window =  modGlassData(window)
nonwindow =  modGlassData(nonwindow)

#t1 = findDependeceT(window,[0,1,2,3,4,5,6,7,8])
#t2 = findDependeceT(nonwindow,[0,1,2,3,4,5,6,7,8])

#1.
#Crossvalidation5DT2(nonwindow,window,[0,1,2,3,4,5,6,7,8])
#Crossvalidation5DT2(window,nonwindow,[0,1,2,3,4,5,6,7,8])

#Crossvalidation5Bayes(window,nonwindow)
#Crossvalidation5Bayes(nonwindow,window)

#Crossvalidation5Bayes(window,nonwindow,t1,t2)
#Crossvalidation5Bayes(nonwindow,window,t2,t1)


#2.

#drawDenpendenceTree(t1)
#drawDenpendenceTree(t2)

#3.
aglass = [180,1.51852,14.09,2.19,1.66,72.67,0.00,9.32,0.00,0.00,6]
aglass = modGlassData([aglass])[0]
'''
result = independentBY(window,nonwindow,aglass)
if result>=0:
    print "it's window glass"
else:
    print "it[s non-window glass"'''

#4
'''result = dependentBY(window,nonwindow,aglass,t1,t2)
if result>=0:
    print "it's window glass"
else:
    print "it's non-window glass"'''

#5
'''root = makedecisiontree(window,nonwindow,[0,1,2,3,4,5,6,7,8])
drawDecideTree(root)
if decide(aglass,root) == "c2":
    print "it's window glass"
else:
    print "it's non-window glass"'''
