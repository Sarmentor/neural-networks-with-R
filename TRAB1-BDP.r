#Mestrado de An�lise de Dados e Sistemas de apoio � Decis�o

#Programa criado para a cadeira de BDP - Trabalho1 - Programa��o em R
#Todas as fun��es foram inicialmente programadas na Linguagem Octave tendo sido
#adaptadas a linguagem R

#Utiliza-se tamb�m uma fun��o fmincg desenvolvida por Carl Edward Rasmussen,
#inicialmente tamb�m obtida na linguagem Octave

#Trabalho realizado por:
#Maria Isabel Moreira
#Rui Sarmento

## NOTA IMPORTANTE - O programa s� foi testado em m�quinas 64-bit, n�o se garante correcto funcionamento em m�quinas 32-bit ##
## Inicializa��o de Vari�veis
rm(list=ls(all=TRUE)) 
directoria <- getwd()
setwd(directoria)
library(R.matlab) #para carregar os ficheiros com dados e parametros para Feedforward que s�o ficheiros .mat 

sigmoid <- function(z){
#SIGMOID Calculo da fun��o sigmoid de z

g = 1.0 / (1.0 + exp(-z))
}

sigmoidGradient <- function(z){
#SIGMOIDGRADIENT returna o gradiente da fun��o sigmoid de z
#   g = SIGMOIDGRADIENT(z) calcula o gradiente da fun��o sigmoide em z 
#   mesmo sendo z um vector ou matriz onde retorna o gradiente para cada elemento.

if(dim(z)[2]){
ncolumn=dim(z)[2]
}else{ncolumn=1}
g = matrix(rep(0),dim(z)[1],ncolumn)
g = (1/(1+ exp(-z)))*(1-(1/(1+exp(-z))))
}

randInitializeWeights <- function(L_in, L_out){
#   W = RANDINITIALIZEWEIGHTS(L_in, L_out) inicializa aleat�riamente os pesos da camada 
#   com L_in liga��es de inputs/entradas e L_out liga��es de outputs/sa�das.
#   Note que W � uma matriz de dimens�es(L_out, 1 + L_in) 
#   uma vez que a primeira linha de W � designada para os termos de bias da rede.
W = matrix(rep(0),L_out, 1 + L_in)
#   Inicializamos W aleatoriamente de modo a quebrar a simetria da rede neuronal, com
#   Valores pequenos gerados aleat�riamente
epsilon_init = 0.12
W = matrix(runif(1:(L_out*(1 + L_in)),-1,1),L_out,1+L_in) * 2 * epsilon_init - epsilon_init
return (W)
}

prediction <- function(Theta1, Theta2, X){
#   p = PREDICTION(Theta1, Theta2, X) retorna a previs�o de sa�da 
#   com os dados de entrada X com os parametros de treino da rede (Theta1, Theta2)

# Inicializa��o de valores �teis
p=0
m = dim(X)[1]
num_labels = dim(Theta2)[1]
ones = rep(1,length(m))
a1=cbind(ones,X)


p=cbind(rep(0,dim(X)[1]))
# c�lculo da sa�da da camada de entrada:
# ver relat�rio com explica��o te�rica para mais informa��o 
h1 = sigmoid(a1 %*% t(Theta1))
h1=cbind(ones,h1)
#sa�da da camada de sa�da 
h2 = sigmoid(h1 %*% t(Theta2))

# ciclo for que c�lcula os valores previstos 
# maximos e retirando o indice que corresponde ao algarismo a prever

for (i in 1: m)
{
h2_linha = h2[i,]
p[i]=which.max(h2_linha[])
}

return (p)

}

nnCostFunction <- function(nn_params,input_layer_length,hidden_layer_length,num_labels,X, y, lambda){
#NNCOSTFUNCTION implementa a fun��o de custo da rede neuronal com duas camadas e que opera problemas de classifica��o
#   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_length, num_labels, ...
#   X, y, lambda) c�lcula o custo e o gradiente da rede neuronal.

#   Os parametros da rede neuronal vem unidimensionalmente vectorizados(nn_params)
#   e precisam de ser convertidos novamente para matrizes de pesos 
# 
#   O parametro retornado grad � um vector com as derivadas partiais da rede neuronal

Theta1 = matrix(nn_params[1:(hidden_layer_length * (input_layer_length + 1))],nrow=hidden_layer_length,ncol=(input_layer_length + 1))

Theta2 = matrix(nn_params[(1 + (hidden_layer_length * (input_layer_length + 1))):length(nn_params)],nrow=num_labels, ncol=(hidden_layer_length + 1))

# Setup de variaveis
m = dim(X)[1]
J = 0
Theta1_grad = matrix(rep(0), dim(Theta1)[1],dim(Theta1)[2])
Theta2_grad = matrix(rep(0), dim(Theta2)[1],dim(Theta2)[2])
  
#         Implementa��o do algoritmo de backpropagation para calculo dos gradientes Theta1_grad and Theta2_grad. 
#         Retorna-se as derivadas parciais da fun��o custo em ordem a Theta1 e Theta2, 
#         os valores s�o introduzidos nas variaveis Theta1_grad e Theta2_grad.
#
#         
#    Nota: O vector y passado para esta fun��o � um vector de classes 
#          contendo valores entre 1..K classes. O vector necessita de ser mapeado
#          num vector bin�rio com 1(uns) e 0(zeros) para ser usado na fun��o 
#          custo da rede neuronal
#    Implementa-se aqui a regulariza��o com o custo e gradientes

delta_2=delta_3=Delta_1=Delta_2=0
ones = rep(1,length(m))

a1 = cbind(ones,X)
a2 = sigmoid( Theta1 %*% t(a1))
a2 = cbind(ones,t(a2))
a3 = sigmoid( Theta2 %*% t(a2))
a3= t(a3)

y=diag(1,num_labels)[y,]
J = 1/m * sum(colSums(-1 * y * log(a3)-(1-y) * log(1-a3))) + (sum(colSums(Theta1[, 2:ncol(Theta1)]^2)) + sum(colSums(Theta2[,2:ncol(Theta2)]^2)))*lambda/m/2

z2=Theta1 %*% t(a1)

delta_3 = a3 - y #(y � a matriz com valores bin�rios para cada uma das classes) 
a=sigmoidGradient(z2)
delta_2 = t(delta_3 %*% Theta2[,2:ncol(Theta2)]) * a 

Delta_2 = Delta_2 + t(delta_3) %*% a2 
Delta_1 = Delta_1 + delta_2 %*% a1

Theta1_grad = (1/m)* Delta_1
Theta2_grad = (1/m)* Delta_2

Theta1_grad[,2:ncol(Theta1_grad)]=Theta1_grad[,2:ncol(Theta1_grad)] + (lambda/m)*Theta1[,2:ncol(Theta1)]
Theta2_grad[,2:ncol(Theta2_grad)]=Theta2_grad[,2:ncol(Theta2_grad)] + (lambda/m)*Theta2[,2:ncol(Theta2)]

#vectoriza��o dos gradientes dos parametros
grad = c(Theta1_grad, Theta2_grad)

#fun��o retorna dois parametros simultaneamente, us�mos listas para tal
ret <- list(J,grad)
names(ret) <- c("J","grad")
return (ret)

}

#a seguinte fun��o permite obter a minimiza��o da fun��o custo avaliando os
#parametros da rede em cada itera��o
fmincg <- function(f, X1, option){
# # Minimize a continuous differentialble multivariate function. Starting point
# # is given by "X1" (D by 1), and the function named in the string "f", must
# # return a function value and a vector of partial derivatives. The Polack-
# # Ribiere flavour of conjugate gradients is used to compute search directions,
# # and a line search using quadratic and cubic polynomial approximations and the
# # Wolfe-Powell stopping criteria is used together with the slope ratio method
# # for guessing initial step sizes. Additionally a bunch of checks are made to
# # make sure that exploration is taking place and that extrapolation will not
# # be unboundedly large. The "length" gives the length of the run: if it is
# # positive, it gives the maximum number of line searches, if negative its
# # absolute gives the maximum allowed number of function evaluations. You can
# # (optionally) give "length" a second component, which will indicate the
# # reduction in function value to be expected in the first line-search (defaults
# # to 1.0). The function returns when either its length is up, or if no further
# # progress can be made (ie, we are at a minimum, or so close that due to
# # numerical problems, we cannot get any closer). If the function terminates
# # within a few iterations, it could be an indication that the function value
# # and derivatives are not consistent (ie, there may be a bug in the
# # implementation of your "f" function). The function returns the found
# # solution "X", a vector of function values "fX" indicating the progress made
# # and "i" the number of iterations (line searches or function evaluations,
# # depending on the sign of "length") used.
# #
# # Usage: [X, fX, i] = fmincg(f, X, option, P1, P2, P3, P4, P5)
# #
# # Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
# #
# #
# # (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
# # 
# # Permission is granted for anyone to copy, use, or modify these
# # programs and accompanying documents for purposes of research or
# # education, provided this copyright notice is retained, and note is
# # made of any changes that have been made.
# # 
# # These programs and documents are distributed without any warranty,
# # express or implied.  As the programs were written for research
# # purposes only, they have not been tested to the degree that would be
# # advisable in any important application.  All use of these programs is
# # entirely at the user's own risk.
# #
# # Altera��es Feitas:
# # 1) Nome da fun��o e especifica��o de argumentos
# # 2) A fun��o original foi obtida na linguagem Octave e teve de ser traduzida para R
# #
# # Read option - L� n�mero de itera��es
if (option) size = option else size = 100



RHO = 0.01                            # a bunch of constants for line searches
SIG = 0.5       # RHO and SIG are the constants in the Wolfe-Powell conditions
INT = 0.1    # don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0                    # extrapolate maximum 3 times the current bracket
MAX = 20                         # max 20 function evaluations per line search
RATIO = 100                                      # maximum allowed slope ratio
realmin = 2.2251^-308
red=1 


i = 0                                            # zero the run size counter
ls_failed = 0                             # no previous line search has failed
fX = NULL ###
#[f1 df1] = eval(argstr)                      # get function value and gradient
aux_fmincg <- costFunction(X1,input_layer_size,hidden_layer_size,num_labels,X, y, lambda)
f1<-aux_fmincg$J
df1 <- aux_fmincg$grad
i = i + (size<0)                                            # count epochs?!
s = -df1                                        # search direction is steepest
d1 = t(-s) %*% s                                                 # this is the slope
z1 = red/(1-d1)                                  # initial step is red/(|s|+1)

while (i < abs(size)){                                      # while not finished
  i = i + (size>0)                                      # count iterations?!

  X0 = X1
  f0 = f1
  df0 = df1                 # make a copy of current values
  X1 = X1 + z1 %*% s                                             # begin line search
  aux_fmincg <- costFunction(X1,input_layer_size,hidden_layer_size,num_labels,X, y, lambda)
  f2<-aux_fmincg$J
  df2 <- aux_fmincg$grad
  i = i + (size<0)                                          # count epochs?!
  d2 = t(df2) %*% s
  f3 = f1 
  d3 = d1 
  z3 = -z1             # initialize point 3 equal to point 1
  if (size>0){
  M = MAX 
  }else{
  M = min(MAX, -size-i)
  }
  success = 0 
  limit = -1                     # initialize quanteties
  while (1){
    while ((f2 > f1+z1*RHO*d1) | (d2 > -SIG*d1) & (M > 0)){
      limit = z1                                         # tighten the bracket
      if (f2 > f1){
        z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3)}                 # quadratic fit
      else{
        A = 6*(f2-f3)/z3+3*(d2+d3)                                 # cubic fit
        B = 3*(f3-f2)-z3*(d3+2*d2)
		#browser()
        z2 = (sqrt(B^2-A*d2*z3^2)-B)/A       # numerical error possible - ok!
      }
      if (is.nan(z2) | is.infinite(z2)){
        z2 = z3/2                  # if we had a numerical problem then bisect
      }
      z2 = max(min(z2, INT*z3),(1-INT)*z3)  # don't accept too close to limits
      z1 = z1 + z2                                           # update the step
      X1 = X1 + z2*s
	  aux_fmincg <- costFunction(X1,input_layer_size,hidden_layer_size,num_labels,X, y, lambda)
      f2<-aux_fmincg$J
      df2 <- aux_fmincg$grad
      M = M - 1 
	  i = i + (size<0)                           # count epochs?!
      d2 = t(df2) %*% s
      z3 = z3-z2                    # z3 is now relative to the location of z2
    }
    if (f2 > f1+z1*RHO*d1 | d2 > -SIG*d1){
      break}                                                # this is a failure
    else if (d2 > SIG*d1){
      success = 1 
	  break}                                             # success
    else if( M == 0){
      break                                                          # failure
    }
	
	
    A = 6*(f2-f3)/z3+3*(d2+d3)                      # make cubic extrapolation
    B = 3*(f3-f2)-z3*(d3+2*d2)
    z2 = -d2*z3*z3/(B+sqrt(B^2-A*d2*z3^2))        # num. error possible - ok!
    if (!is.real(z2) | is.nan(z2) | is.infinite(z2) | z2 < 0 ){  # num prob or wrong sign?
                                         
                  
      
    if ((limit > -0.5) & (z2+z1 > limit))          # extraplation beyond max?
      z2 = (limit-z1)/2                                               # bisect
    else if ((limit < -0.5) & (z2+z1 > z1*EXT))       # extrapolation beyond limit
      z2 = z1*(EXT-1.0)	  # set to extrapolation limit
	  
    else if (z2 < -z3*INT)
      z2 = -z3*INT
    else if ((limit > -0.5) & (z2 < (limit-z1)*(1.0-INT)))   # too close to limit?
      z2 = (limit-z1)*(1.0-INT)
	else if (limit < -0.5) # if we have no upper limit the extrapolate the maximum amount otherwise bisect
	  z2 = z1 * (EXT-1) 
	else if (limit > -0.5)
	  z2 = (limit-z1)/2
	 
	}
    f3 = f2 
	d3 = d2 
	z3 = -z2                  # set point 3 equal to point 2
    z1 = z1 + z2 
	X1 = X1 + z2*s                      # update current estimates
	aux_fmincg <- costFunction(X1,input_layer_size,hidden_layer_size,num_labels,X, y, lambda)
    f2<-aux_fmincg$J
    df2 <- aux_fmincg$grad
    M = M - 1 
	i = i + (size<0)                             # count epochs?!
    d2 = t(df2) %*% s
  }
  
                                                      # end of line search

  if (success){                                         # if line search succeeded
    f1 = f2
	if (!is.null(fX)) fX=t(cbind(t(fX), f1)) else fX = t(f1)
    print(paste('Itera��o: ',i,'| Custo:',  f1))
	
    s = (t(df2) %*% df2-t(df1) %*% df2)%/%(t(df1) %*% df1) %*% s - df2      # Polack-Ribiere direction
    tmp = df1 
	df1 = df2 
	df2 = tmp 	# swap derivatives
    
	d2 = t(matrix(df1,1,length(df1))) %*% s
    if (any(d2 > 0)){                                      # new slope must be negative
      s = -df1                              # otherwise use steepest direction
      d2 = t(-s) %*% s    
    }
    z1 = z1 * min(RATIO, d1/(d2-realmin))          # slope ratio but max RATIO
    d1 = d2
    ls_failed = 0                              # this line search did not fail
  }else{
    X1 = X0 
	f1 = f0 
	df1 = df0  # restore point from before failed line search
    if (ls_failed | i > abs(size)){          # line search failed twice in a row
      break                             # or we ran out of time, so we give up
    }
    tmp = df1 
	df1 = df2 
	df2 = tmp                         # swap derivatives
    s = -df1                                                    # try steepest
    d1 = t(-s) %*% s
    z1 = 1/(1-d1)                     
    ls_failed = 1                                    # this line search failed
  }
  
 
}
ret <- list(X1,fX,i)
names(ret)<- c("X","fX","i")
return (ret)
}


## Setup de variaveis
input_layer_size  = 400   # Imagens de digitos manuscritos com 20x20 pixeis cada uma
hidden_layer_size = 25    # 25 camadas ocultas
num_labels = 10           # 10 classes diferentes, de 1 a 10 (algarismos de 0 a 9)  
                          # (note que "0" foi mapeado com 10)


# Carregar os dados fornecidos
print('Carregando dados ...')

dados <- readMat('data.mat') 
X <- dados$X
y <- dados$y
m = length(X)[1]

#print('Program browser()d. Escreva cont e carregue enter para continuar.') 
#browser() 

# Carregamos os dados dos parametros da rede pr�-determinados para a fase de Feedforward

print('Carregando os parametros da rede neuronal ...')

# Carrega os pesos para  Theta1 and Theta2
dados_thetas <- readMat('weights.mat') 
Theta1 <- dados_thetas$Theta1
Theta2 <- dados_thetas$Theta2

# vectoriza��o dos parametros Theta 
nn_params = c(Theta1, Theta2)

## ================ Part 3: Compute Cost (Feedforward) ================
#  Vamos come�ar por implementar o feedforward da rede neuronal 
#  com os dados fornecidos sendo que a fun��o nnCostFunction retornar� o custo associado.
#
print('Feedforward com a rede neuronal...')

# Parametro de regulariza��o lambda a 0
 lambda = 0 

 aux_cost = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lambda) 
 J = aux_cost$J
print(paste('Custo com parametros carregados de weights.m sem regulariza��o : ',J,'(este valor deve ser aproximadamente 0.287629)')) 

print('Verificando Fun��o custo agora com regulariza��o (lambda=1)')

#set  parametro de regulariza��o
lambda = 1 

aux_cost = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda) 
#J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda) 
J = aux_cost$J

print(paste('Custo com parametros carregados de weights.m com regulariza��o:' , J,'(este valor deve ser aproximadamente 0.383770)')) 

#print('Program browser()d. Escreva cont e carregue enter para continuar.') 
#browser() 


print('Avaliando o gradiente sigmoide...')

g = sigmoidGradient(matrix(c(1, -0.5, 0, 0.5, 1),1)) 
print('Sigmoid gradient avaliado para [1 -0.5 0 0.5 1]:  ') 
print(g)

#print('Program browser()d. Escreva cont e carregue enter para continuar.') 
#browser() 

#Avaliando as previs�es com feedforward e os parametros previamente fornecidos
pred = prediction(Theta1, Theta2, X)
iguais <- pred==y
precisao <- (sum(iguais==TRUE)/dim(y)[1])*100

print(paste('Precis�o do Treino com Feedforward: ', precisao,'% com lambda = ',lambda)) 

# Implementa��o de rede neuronal com duas camadas utilizando
# a inicializa��o dos parametros Theta aleatoriamente
# com a fun��o (randInitializeWeights.m)

print('Inicializando os parametros da rede neuronal aleatoriamente...')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size) 
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels) 

# Vectoriza parametros
initial_nn_params = c(initial_Theta1, initial_Theta2) 


#  Come�a aqui a implementa��o do algoritmo de backpropagation da rede neuronal.

print('Avaliando o algoritmo de Backpropagation com regulariza��o... ')

lambda = 3 

# Output dos valores da Fun��o Custo para verifica��o 
aux_cost  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda) 
debug_J = aux_cost$J

print(paste('Custo com lambda = 3:', debug_J,'(este valor deve ser aproximadamente 0.576051)')) 

#print('Program browser()d. Escreva cont e carregue enter para continuar.') 
#browser() 


# Para treinar a rede neuronal us�mos a fun��o fmincg. 
# Para tal temos que lhe fornecer os c�lculos dos gradientes
# da fun��o custo

print('A treinar a Rede Neuronal (pode demorar alguns minutos)... ')

#  N�mero de itera��es para c�lculo dos parametros de treino da rede neuronal
option = 400 #n�mero de itera��es para atingir n�veis aproximados � precis�o atingida com parametros fornecidos para feedforward 
#  Agora com lambda = 1
lambda = 1 

# Crea��o de "short hand" para que a fun��o custo seja minimizada - a rever necessidade
costFunction <- function(nn_params,...)
{
nnCostFunction(nn_params, ...) 
} 
aux <- fmincg(costFunction(initial_nn_params,input_layer_size, hidden_layer_size, num_labels, X, y, lambda), initial_nn_params, option) 
nn_params = aux$X
cost = aux$fX

# Obter-se Theta1 and Theta2 devolvidos ap�s 
# minimiza��o do custo em forma de matriz
Theta1 = matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))], nrow=hidden_layer_size, ncol=(input_layer_size + 1))  
Theta2 = matrix(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))):ncol(nn_params)], nrow=num_labels,ncol= (hidden_layer_size + 1)) 

#print('Programa browser()d. Escreva cont e carregue enter para continuar.') 
#browser() 


#  Depois do treino vamos prever as classes para cada
#  exemplo com a fun��o prediction. Assim, pudemos c�lcular 
#  a precis�o da previs�o ap�s treino da rede

pred = prediction(Theta1, Theta2, X)
iguais <- pred==y
precisao <- (sum(iguais==TRUE)/dim(y)[1])*100

print(paste('Precis�o do Treino: ', precisao , '% para ',option,' itera��es e com lambda = ',lambda)) 


