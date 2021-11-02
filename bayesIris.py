import numpy as np
from sklearn import datasets
import math

np.set_printoptions(suppress=True)

# Media di una sequenza di numeri
def media(nums):
	return sum(nums)/float(len(nums))

# Calcolo elemento ij di una matrice di covarianza
def sigmaIJ(i,j,pattern,mediaI,mediaJ):
	count = 0.0
	for x in pattern:
		count = count + ((x[i]-mediaI)*(x[j]-mediaJ))
		sigmaIJ = count / float(len(pattern))
	return sigmaIJ

# Calcola matrice covarianza di un dataset N dimensionale
def covarianza(dataset,vettoreMedio):
	cov = [[0.0 for x in range(len(dataset[0]))] for y in range(len(dataset[0]))] 
	for i in range(len(dataset[0])):
		for j in range(len(dataset[0])):
			cov[i][j] = sigmaIJ(i,j,dataset,vettoreMedio[i],vettoreMedio[j])
	return cov

# Divido le Nuple del dataset in base alla classe
def splitDatasetByClass(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		classLabel = int(vector[-1])
		vector = vector[:-1] #TOLGO LA Y DAL DATASET
		if (classLabel not in separated):
			separated[classLabel] = list()
		separated[classLabel].append(vector)
	return separated

# Ritorno un vettore medio per ogni classe del dataset
def vettoriMedi(mapData):
	vect = dict()
	for label,rows in mapData.items():
		vect[label] = np.array( [(media(column)) for column in zip(*rows)] )
	return vect

# Ritorno una matrice di covarianza per ogni classe del dataset
def matriciCovarianza(mapData,vettoriMedi):
	matr = dict()
	for label,rows in mapData.items():
		matr[label] = np.array(covarianza(rows,vettoriMedi[label]))
		#OPPURE
		#matr[label] = np.cov(np.array(rows).T,bias=True)
	return matr


def probabilitaPriori(datasetLength,mapData):
	probPriori = dict()
	for label,rows in mapData.items():
		probPriori[label] = float(len(rows))/float(datasetLength)
	return probPriori

def multinomialDistribution(input_X,cov,med):
	v = input_X - med
	d = len(med)
	#Denominatore
	a = math.pow((2*math.pi),d/2)
	b = math.pow(np.linalg.det(cov),(0.5))
	#Esponente
	dot = np.dot(np.dot(v,np.linalg.inv(cov)),v)
	exp = -((dot)/2.0)	
	p = (1.0/(a*b)) * math.pow(math.e,exp)
	return p

def probabilitaCondizionate(input_x,class_labels,vettoriMedi,covarianze):
	probCondizionate = dict()
	for label in class_labels:
		probCondizionate[label] = multinomialDistribution(input_x,vettoriMedi[label],covarianze[label])
	return probCondizionate

def teoremaBayes(pCond,pPriori,pAbs):
	p = dict()
	for label in probCondizionate.keys():
		p[label] = (pCond[label]*pPriori[label])/pAbs
	return p



#Leggo dataset
iris = datasets.load_iris()

X = iris.data
Y = iris.target
XY = np.append(X,np.array([Y]).T,axis=1)

#Training set (Righe 0-29 , 50-79 , 109-129)
XT = np.append(XY[:29,:],XY[50:79,:],axis=0)
XT = np.append(XT,XY[100:129,:],axis=0)
#Test set (Righe 30-49 , 80-99, 130-149)
XTest = np.append(XY[30:49,:],XY[80:99,:],axis=0)
XTest = np.append(XTest,XY[130:149,:],axis=0)


####################### CALCOLO PARAMETRI COL TRAINING SET #########################


#Creo un dict del tipo {class_Y} ----> [Nuple di quella classe senza la feature della Y]
mapData = splitDatasetByClass(XT)
#Creo un vettore medio per ognuna delle classi del training set
#E' un Dict: {class_Y} ---> [vettore medio Nuple della classe Y]
vettoriMedi = vettoriMedi(mapData)
#Stampa vettori medi
for label,vector in vettoriMedi.items():
	print("\tVettore medio per la classe "+str(label))
	print(vettoriMedi[label])
print("")
#Creo una matrice di covarianza per ognuna delle classi del training set
#E' un Dict: {class_Y} ---> [matrice covarianza della classe Y]
matriciCovarianza = matriciCovarianza(mapData,vettoriMedi)
#Stampa matrici covarianza
for label,matrix in matriciCovarianza.items():
	print("\tCovarianza per la classe "+str(label))
	print(matriciCovarianza[label])
print("")

######################## CALCOLO PROB PRIORI DELL'INTERO DATASET ##########################


#Stima delle probabilita' a priori di pescare un elemento di una certa classe, ogni classe ha la sua prob priori
#	[Class_Y] -> [P(Wi) della classe Y]
probPriori = probabilitaPriori(len(X),mapData)

#Inizializzo matrice confusione
numOfClass = len(mapData.keys())
confusionMatrix = [ [ 0 for i in range(numOfClass) ] for j in range(numOfClass) ]

###################################### TEST SET ############################################
print("\t\tClassificazione del test set")
predizioniCorrette = 0
for input_xy in XTest:
	Y_input = input_xy[-1] 
	input_x = input_xy[:-1] # tolgo la Y

	"""  
	Stimo le probabilita' condizionate che dato un pattern appartenente alla 
	classe W, sia proprio il pattern X in input. ogni classe ha la sua prob cond."""

	#[Class_Y] -> [P(input_x/Wi)]
	probCondizionate = probabilitaCondizionate(input_x,mapData.keys(),matriciCovarianza,vettoriMedi)

	#Calcolo probabilita' assoluta come prodotto probPriori e probCondizionate
	probAssoluta = 0.0
	for label in probCondizionate.keys():
		probAssoluta = probAssoluta + (probCondizionate[label]*probPriori[label])

	#CALCOLO PROB A POSTERIORI CHE DATO UN PATTERN X -> X APPARTIENE ALLA CLASSE W
	p = teoremaBayes(probCondizionate,probPriori,probAssoluta)
	#La y predetta è la Y avente la P(Y/X) più alta
	Y_predicted =  sorted(p, key=p.get, reverse=True)[0]

	#Stampa del risultato del pattern corrente
	predizioniCorrette = predizioniCorrette+int((Y_predicted == Y_input))
	print("X="+str(input_x)+" Y=["+str(int(Y_input))+"]  --->  Classe predetta = ["+str(Y_predicted)+"]  ("+str((Y_predicted == Y_input))+")")

	#Incremento matrice confusione
	confusionMatrix[int(Y_input)][int(Y_predicted)]=confusionMatrix[int(Y_input)][int(Y_predicted)]+1
    
#Stampa matrice confusione
print("\n------------------------Matrice confusione------------------------\n")
startRow = "\t\t "
for i in range(len(confusionMatrix)):
	startRow=startRow+"Pred_"+str(i)+"\t\t"
print(startRow+"\n")
for i in range(len(confusionMatrix)):
	row = "Real_"+str(i)+"\t  \t "
	for j in range(len(confusionMatrix)):
		row = row+str(confusionMatrix[i][j])+"\t\t"
	print(row+"\n")
print("------------------------------------------------------------------\n")
#Stampa efficienza totale del classificatore
print("\tEfficienza classificatore : "+ str((float(predizioniCorrette)/float(len(XTest)))*100.0)+"%")