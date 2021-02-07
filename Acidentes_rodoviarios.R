#*****************************************************************************
# Aprendizado conforme modelo apresentado no curso de Power BI com Machine Learning, da DSA.
#       
#*****************************************************************************

# Definindo a pasta de trabalho
setwd("E:/Users/Vander/Desktop/PUC/TCC - Trabalho de Conclus„o de Curso/Bases")
getwd()


# Instalando os pacotes para o projeto 
# Obs: os pacotes precisam ser instalados apenas uma vez
install.packages("Amelia")
install.packages("caret")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("reshape")
install.packages("randomForest")
install.packages("e1071")
install.packages("DMwR")


# Carregando os pacotes 
library(Amelia)
library(ggplot2)
library(caret)
library(reshape)
library(randomForest)
library(dplyr)
library(e1071)
library(graphics)
library(utils)
library(DMwR)

# Carregando o dataset
dados_acidentes <- read.csv("dados/dataset_amostral.csv")

# Visualizando os dados e sua estrutura
View(dados_acidentes)
dim(dados_acidentes)
str(dados_acidentes) 
summary(dados_acidentes)

# Convertendo dados para utf-8

dados_acidentes$tipo_acidente <- iconv(dados_acidentes$tipo_acidente, to = "latin1//TRANSLIT", from= "UTF-8")
dados_acidentes$causa_acidente <- iconv(dados_acidentes$causa_acidente, to = "latin1//TRANSLIT", from= "UTF-8")
dados_acidentes$tracado_via <- iconv(dados_acidentes$tracado_via, to = "latin1//TRANSLIT", from= "UTF-8")
dados_acidentes$tipo_pista <- iconv(dados_acidentes$tipo_pista, to = "latin1//TRANSLIT", from= "UTF-8")
dados_acidentes$fase_dia <- iconv(dados_acidentes$fase_dia, to = "latin1//TRANSLIT", from= "UTF-8")
dados_acidentes$condicao_metereologica <- iconv(dados_acidentes$condicao_metereologica, to = "latin1//TRANSLIT", from= "UTF-8")

#################### An·lise exploratÛria, limpeza e transformaÁ„o ####################


View(dados_acidentes)


# Verificando valores ausentes e removendo do dataset
sapply(dados_acidentes, function(x) sum(is.na(x)))
missmap(dados_acidentes, main = "Valores Missing Observados")
dados_acidentes <- na.omit(dados_acidentes)


# para fatores (categorias)
str(dados_acidentes) 


# Convertendo vari·veis para tipo fator

dados_acidentes$br <- as.factor(dados_acidentes$br)
dados_acidentes$km <- as.factor(dados_acidentes$km)
dados_acidentes$tipo_acidente <- as.factor(dados_acidentes$tipo_acidente)
dados_acidentes$tipo_pista <- as.factor(dados_acidentes$tipo_pista)
dados_acidentes$causa_acidente <- as.factor(dados_acidentes$causa_acidente)
dados_acidentes$tracado_via <- as.factor(dados_acidentes$tracado_via)
dados_acidentes$fase_dia <- as.factor(dados_acidentes$fase_dia)
dados_acidentes$condicao_metereologica <- as.factor(dados_acidentes$condicao_metereologica)


# Dataset apÛs as conversıes.
str(dados_acidentes) 
sapply(dados_acidentes, function(x) sum(is.na(x)))
missmap(dados_acidentes, main = "Valores Missing Observados")
dados_acidentes <- na.omit(dados_acidentes)
missmap(dados_acidentes, main = "Valores Missing Observados")
dim(dados_acidentes)
View(dados_acidentes)


# A vari·vel alvo que foi escolhida foi o tipo_pista.
?table
table(dados_acidentes$tipo_pista)

# Vejamos as porcentagens entre as classes
prop.table(table(dados_acidentes$tipo_pista))

# Plot da distribuiÁ„o usando ggplot2
qplot(tipo_pista, data = dados_acidentes, geom = "bar") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Set seed
set.seed(12345)

# Amostragem estratificada. 
# Seleciona as linhas de acordo com a vari·vel ipo_pista
?createDataPartition
indice <- createDataPartition(dados_acidentes$tipo_pista, p = 0.75, list = FALSE)
dim(indice)

# Definimos os dados de treinamento como subconjunto do conjunto de dados original
# com n˙mero de indice de linha (conforme identificado acima) e todas as colunas
dados_treino <- dados_acidentes[indice,]
dim(dados_treino)
table(dados_treino$tipo_pista)

# Veja as porcentagens entre as classes
prop.table(table(dados_treino$tipo_pista))

# N√˙mero de registros no dataset de treinamento
dim(dados_treino)

# Comparamoos as porcentagens entre as classes de treinamento e dados originais
compara_dados <- cbind(prop.table(table(dados_treino$tipo_pista)), 
                       prop.table(table(dados_acidentes$tipo_pista)))
colnames(compara_dados) <- c("Treinamento", "Original")
compara_dados

# Melt Data - Converte colunas em linhas
?reshape2::melt
melt_compara_dados <- melt(compara_dados)
melt_compara_dados

# Plot para ver a distribui√ß√£o do treinamento vs original
ggplot(melt_compara_dados, aes(x = X1, y = value)) + 
  geom_bar( aes(fill = X2), stat = "identity", position = "dodge") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Tudo o que n„o est· no dataset de treinamento est·° no dataset de teste. Observe o sinal - (menos)
dados_teste <- dados_acidentes[-indice,]
dim(dados_teste)
dim(dados_treino)

#################### Modelo de Machine Learning ####################

# Construindo a primeira vers„o do modelo
?randomForest
View(dados_treino)
modelo_v1 <- randomForest(tipo_pista ~ condicao_metereologica + tracado_via + pessoas + mortos + feridos_leves 
                          + feridos_graves + ilesos + ignorados + feridos + veiculos, data = dados_treino)
modelo_v1

# Avaliando o modelo
plot(modelo_v1)

# Previsıes com dados de teste
previsoes_v1 <- predict(modelo_v1, dados_teste)

# Confusion Matrix
?caret::confusionMatrix
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$tipo_pista, positive = NULL)
cm_v1

# Calculando Precision, Recall e F1-Score, MÈtricas de avalia√ß√£o do modelo preditivo
y <- dados_teste$tipo_pista
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1, y)
precision
?posPredValue

recall <- sensitivity(y_pred_v1, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

# Balanceamento de classe
?SMOTE

# Aplicando o SMOTE - SMOTE: Synthetic Minority Over-sampling Technique
# https://arxiv.org/pdf/1106.1813.pdf
table(dados_treino$tipo_pista)
prop.table(table(dados_treino$tipo_pista))
set.seed(9560)
dados_treino_bal <- SMOTE(tipo_pista ~ ., data  = dados_treino)                         
table(dados_treino_bal$tipo_pista)
prop.table(table(dados_treino_bal$tipo_pista))

# Construindo a segunda vers„o do modelo
modelo_v2 <- randomForest(tipo_pista ~ condicao_metereologica + tracado_via + pessoas + mortos + feridos_leves 
                          + feridos_graves + ilesos + ignorados + feridos + veiculos, data = dados_treino_bal)
modelo_v2

# Avaliando o modelo
plot(modelo_v2)

# Previsıes com dados de teste
previsoes_v2 <- predict(modelo_v2, dados_teste)

# Confusion Matrix
?caret::confusionMatrix
cm_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$tipo_pista, positive = NULL)
cm_v2

# Calculando Precision, Recall e F1-Score, MÈtricas de avalia√ß√£o do modelo preditivo
y <- dados_teste$tipo_pista
y_pred_v2 <- previsoes_v2

precision <- posPredValue(y_pred_v2, y)
precision

recall <- sensitivity(y_pred_v2, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

# Import‚ncia das vari·veis preditoras para as previsıes
View(dados_treino_bal)
varImpPlot(modelo_v2)

# Obtendo as vari·veis mais importantes
imp_var <- importance(modelo_v2)
varImportance <- data.frame(Variables = row.names(imp_var), 
                            Importance = round(imp_var[ ,'MeanDecreaseGini'],2))

# Criando o rank de vari√°veis baseado na import√¢ncia
rankImportance <- varImportance %>% 
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))

# Usando ggplot2 para visualizar a import√¢ncia relativa das vari·veis
ggplot(rankImportance, 
       aes(x = reorder(Variables, Importance), 
           y = Importance, 
           fill = Importance)) + 
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank), 
            hjust = 0, 
            vjust = 0.55, 
            size = 4, 
            colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() 

# Construindo a terceira vers„o do modelo apenas com as vari·veis mais importantes
colnames(dados_treino_bal)
modelo_v3 <- randomForest(tipo_pista ~ condicao_metereologica + tracado_via + mortos + feridos_graves 
                          + ilesos + feridos + veiculos, data = dados_treino_bal)
modelo_v3

# Avaliando o modelo
plot(modelo_v3)

# Previsıes com dados de teste
previsoes_v3 <- predict(modelo_v3, dados_teste)

# Confusion Matrix
?caret::confusionMatrix
cm_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$tipo_pista, positive = NULL)
cm_v3

# Calculando Precision, Recall e F1-Score, MÈtricas de avalia√ß√£o do modelo preditivo
y <- dados_teste$tipo_pista
y_pred_v3 <- previsoes_v3

precision <- posPredValue(y_pred_v3, y)
precision

recall <- sensitivity(y_pred_v3, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

# Salvando o modelo em disco
saveRDS(modelo_v3, file = "modelo/modelo_v3.rds")

# Carregando o modelo
modelo_final <- readRDS("modelo/modelo_v3.rds")
modelo_final


