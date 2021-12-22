import pdb
import c45

c1 = c45.C45("/home/d19125691/Documents/Experiments/ontologyDCQ/onto-DCQ-FS/datasets/iris/iris.data",
         "/home/d19125691/Documents/Experiments/ontologyDCQ/onto-DCQ-FS/datasets/iris/iris1.names")
c1.fetchData()
c1.preprocessData()
c1.generateTree()
c1.printTree()