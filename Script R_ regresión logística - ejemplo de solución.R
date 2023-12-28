
library(car)
library(ggpubr)
library(lmtest)
library(tidyverse)

################################################################################
# Vamos a construir un modelo de regresión logística para predecir la variable
# EN, de acuerdo con las siguientes instrucciones:
# 1. Definir la semilla a utilizar, que corresponde a los últimos cuatro dígitos
#    del RUN (sin considerar el dígito verificador) del integrante de mayor edad
#    del equipo.
# 2. Seleccionar una muestra de 90 mujeres (si la semilla es un número par) o
#    90 hombres (si la semilla es impar), asegurando que la mitad tenga estado
#    nutricional “sobrepeso” y la otra mitad “no sobrepeso”. Dividir esta
#    muestra en dos conjuntos: los datos de 60 personas (30 con EN “sobrepeso”)
#    para utilizar en la construcción de los modelos y 30 personas (15 con EN
#    “sobrepeso”) para poder evaluarlos.
################################################################################

# Fijamos la carpeta de trabajo
setwd("~/Downloads")

# Cargamos los datos
datos <- read.csv2("Datos Heinz et al 2003.csv")
datos.ext <- datos %>% mutate(IMC = Weight / (Height / 100)**2) %>%
  mutate(EN = ifelse(IMC < 25, "no sobrepeso", "sobrepeso"))
datos.ext[["Gender"]] <- factor(datos.ext[["Gender"]])
datos.ext[["EN"]] <- factor(datos.ext[["EN"]])
datos.ext[["Id"]] <- 1:nrow(datos.ext) # Para revisar independencia de las muestras

# Fijamos la semilla
set.seed(1111)

# Obtenemos la muestra
muestra.a <- datos.ext %>% filter(Gender == 1 & EN == "no sobrepeso") %>%
  sample_n(45, replace = FALSE)
muestra.b <- datos.ext %>% filter(Gender == 1 & EN == "sobrepeso") %>%
  sample_n(45, replace = FALSE)

i.train <- sample(1:45, 30)
muestra.train <- rbind(muestra.a[i.train, ], muestra.b[i.train, ])
muestra.test <- rbind(muestra.a[-i.train, ], muestra.b[-i.train, ])

# Verificamos que no cometimos algún error con las muestras
stopifnot(all(muestra.train$Id == unique(muestra.train$Id)))
stopifnot(all(muestra.test$Id == unique(muestra.test$Id)))
stopifnot(!any(muestra.train$Id %in% muestra.test))

# Vamos a desordenar la muestra para que no queden ordenados los grupos
muestra.train <- muestra.train[sample(1:nrow(muestra.train)), ]
muestra.test <- muestra.test[sample(1:nrow(muestra.test)), ]

################################################################################
# 3. Recordar las ocho posibles variables predictoras seleccionadas de forma
#    aleatoria en el ejercicio anterior.
################################################################################

pred.pos.noms <-c("Knee.Girth", "Bicep.Girth", "Ankles.diameter", "Chest.depth",
                  "Shoulder.Girth", "Navel.Girth", "Hip.Girth", "Biiliac.diameter")

################################################################################
# 4. Seleccionar, de las otras variables, una que el equipo considere que podría
#    ser útil para predecir la clase EN, justificando bien esta selección.
# 5. Usando el entorno R y paquetes estándares, construir un modelo de
#    regresión logística con el predictor seleccionado en el paso anterior y
#    utilizando de la muestra obtenida.
################################################################################

# Vamos a elegir la variable peso como predictor para el modelo de RLogS, pues
# esta variable está fuertemente relacionada con el IMC, que a su vez se usa
# como indicador para determinar el sobrepeso.

rlogs <- glm(EN ~ Weight, data = muestra.train,
             family = binomial(link = "logit"))

cat("\nModelo de regresión logística simple\n")
print(summary(rlogs))

################################################################################
# 6. Usando herramientas estándares para la exploración de modelos del entorno
#    R, buscar entre dos y cinco predictores de entre las variables
#    seleccionadas al azar, recordadas en el punto 3, para agregar al modelo
#    obtenido en el paso 5.
################################################################################

# Podemos buscar un predictor (desde los elegidos aleatoriamente como posibles)
# que nos ayude a mejorar el modelo de RLogS usando llamadas reiteradas a la
# función add1(), por ejemplo:
cat("\n")
print(add1(rlogs, scope = pred.pos.noms))

# O podríamos crear un modelo máximo con todas los posibles predictores elegidos
# y usar llamadas reiteradas a la función drop1(), por ejemplo:
fmla <- formula(paste(c(". ~ .", pred.pos.noms), collapse = " + "))
rlogm.max <- update(rlogs, fmla)
print(drop1(rlogm.max))

# O podemos hacer que R busque un modelo entre nuestro modelo de RLogS y el
# modelo máximo:
print(rlogm <- step(rlogs, scope = list(lower = rlogs, upper = rlogm.max),
                    direction = "both"))

################################################################################
# 7. Evaluar la confiabilidad de los modelos (i.e. que tengan un buen nivel de
#    ajuste y son generalizables) y “arreglarlos” en caso de que tengan algún
#    problema.
################################################################################

# Evaluemos primero el modelo de RLogS

# Primero, revisemos el ajuste
cat("\nModelo de RLogS encontrado:")
print(summary(rlogs))

# Podemos ver que utilizar el peso para predecir el estado nutricional, tal y
# como sospechábamos, conseguimos una importante reducción de "devianza", lo
# que nos indica que el modelo  consigue un buen ajuste.

# Ahora revisemos los residuos y estadísticas de influencia de los casos
eval.rlogs <- data.frame(standardized.residuals = rstandard(rlogs))
eval.rlogs[["studentized.residuals"]] <-rstudent(rlogs)
eval.rlogs[["cooks.distance"]] <- cooks.distance(rlogs)
eval.rlogs[["dfbeta"]] <- dfbeta(rlogs)
eval.rlogs[["dffit"]] <- dffits(rlogs)
eval.rlogs[["leverage"]] <- hatvalues(rlogs)
eval.rlogs[["covariance.ratios"]] <- covratio(rlogs)

cat("\nInfluencia de los casos:\n")
cat("------------------------\n")

# 95% de los residuos estandarizados deberían estar entre −1.96 y +1.96, y 99%
# entre -2.58 y +2.58.
sospechosos1 <- which(abs(eval.rlogs[["standardized.residuals"]]) > 1.96)
cat("- Residuos estandarizados fuera del 95% esperado: ")
print(sospechosos1)

# Observaciones con distancia de Cook mayor a uno.
sospechosos2 <- which(eval.rlogs[["cooks.distance"]] > 1)
cat("- Residuos con distancia de Cook mayor que 1: ")
print(sospechosos2)

# Observaciones con apalancamiento superior al doble del apalancamiento
# promedio: (k + 1)/n.
apalancamiento.promedio <- 2 / nrow(muestra.train)
sospechosos3 <- which(eval.rlogs[["leverage"]] > 2 * apalancamiento.promedio)

cat("- Residuos con apalancamiento fuera de rango (promedio = ",
    apalancamiento.promedio, "): ", sep = "")

print(sospechosos3)

# DFBeta debería ser < 1.
sospechosos4 <- which(apply(eval.rlogs[["dfbeta"]] >= 1, 1, any))
names(sospechosos4) <- NULL
cat("- Residuos con DFBeta mayor que 1: ")
print(sospechosos4)

# Finalmente, los casos no deberían desviarse significativamente
# de los límites recomendados para la razón de covarianza:
# CVRi mayor que 1 + [3(k + 1)/n]
# CVRi menor que 1 – [3(k + 1)/n]
CVRi.lower <- 1 - 3 * apalancamiento.promedio
CVRi.upper <- 1 + 3 * apalancamiento.promedio
sospechosos5 <- which(eval.rlogs[["covariance.ratios"]] < CVRi.lower |
                        eval.rlogs[["covariance.ratios"]] > CVRi.upper)
cat("- Residuos con razón de covarianza fuera de rango ([", CVRi.lower, ", ",
    CVRi.upper, "]): ", sep = "")

print(sospechosos5)

sospechosos <- c(sospechosos1, sospechosos2, sospechosos3, sospechosos4,
                 sospechosos5)

sospechosos <- sort(unique(sospechosos))
cat("\nResumen de observaciones sospechosas:\n")

print(round(eval.rlogs[sospechosos,
                     c("cooks.distance", "leverage", "covariance.ratios")],
            3))

# Si bien hay algunas observaciones que podrían considerarse atípicas, la
# distancia de Cook para todas ellas se aleja bastante de 1, por lo que no
# deberían ser causa de preocupación.

# Ahora verifiquemos el supuesto de linealidad de la relación
xs1 <- data.frame(Logit = log(fitted(rlogs)/(1-fitted(rlogs))),
                  Weight = muestra.train[["Weight"]])
pxs1 <- ggscatter(data = xs1, x = "Logit", y = "Weight", conf.int = TRUE)
print(pxs1)

# Vemos que es perfectamente lineal !!

# Revisemos ahora los residuos y si cumplen las condiciones necesarias
xs2 <- data.frame(Indice = 1:nrow(muestra.train),
                  Residuo.estandarizado = rstandard(rlogs))
pxs2 <- ggscatter(data = xs2, x = "Indice", y = "Residuo.estandarizado")
print(pxs2)

# Vemos que no aparece un patrón claro ni una variación consistente de la
# varianza, lo que podemos confirmar con pruebas auxiliares:
print(shapiro.test(resid(rlogs)))
print(bptest(rlogs))

# Cumple bien con estos requisitos.

# Finalmente, revisamos que los residuos sean independientes
cat("\nPrueba de Durbin y Watson:\n")
print(durbinWatsonTest(rlogs))

# Vemos que no hay razones para sospechar que los residuos no sean independientes

# Así, el modelo de RLogS parece tener un buen ajuste y cumplir bien con las
# condiciones para ser confiable.



# Veamos ahora el modelo de RLogM.

# Primero, revisemos el ajuste
cat("\nModelo de RLogM encontrado:")
print(summary(rlogm))

# Podemos ver que el modelo consigue una importante reducción de la "devianza",
# indicando que se consigue un buen ajuste. Sin embargo, es preocupante que,
# ahora, la variable peso aparezca como poco relevante para el modelo. Debemos
# estar atentos a problemas de multicolinealidad. Evaluemos esta condición.
cat("\nFactores de inflación de la varianza:\n")
print(vif(rlogm))
print(1 / vif(rlogm))

# Podemos notar que todos los factores de inflación de la varianza son lejanos
# al límite de 10 y ninguna tolerancia es menos a 0.2, lo que indicaría que no
# hay presencia de multicolinealidad.

# Ahora revisemos los residuos y estadísticas de influencia de los casos
eval.rlogm <- data.frame(standardized.residuals = rstandard(rlogm))
eval.rlogm[["studentized.residuals"]] <-rstudent(rlogm)
eval.rlogm[["cooks.distance"]] <- cooks.distance(rlogm)
eval.rlogm[["dfbeta"]] <- dfbeta(rlogm)
eval.rlogm[["dffit"]] <- dffits(rlogm)
eval.rlogm[["leverage"]] <- hatvalues(rlogm)
eval.rlogm[["covariance.ratios"]] <- covratio(rlogm)

cat("\nInfluencia de los casos:\n")
cat("------------------------\n")

# 95% de los residuos estandarizados deberían estar entre −1.96 y +1.96, y 99%
# entre -2.58 y +2.58.
sospechosos1 <- which(abs(eval.rlogm[["standardized.residuals"]]) > 1.96)
cat("- Residuos estandarizados fuera del 95% esperado: ")
print(sospechosos1)

# Observaciones con distancia de Cook mayor a uno.
sospechosos2 <- which(eval.rlogm[["cooks.distance"]] > 1)
cat("- Residuos con distancia de Cook mayor que 1: ")
print(sospechosos2)

# Observaciones con apalancamiento superior al doble del apalancamiento
# promedio: (k + 1)/n.
apalancamiento.promedio <- 2 / nrow(muestra.train)
sospechosos3 <- which(eval.rlogm[["leverage"]] > 2 * apalancamiento.promedio)

cat("- Residuos con apalancamiento fuera de rango (promedio = ",
    apalancamiento.promedio, "): ", sep = "")

print(sospechosos3)

# DFBeta debería ser < 1.
sospechosos4 <- which(apply(eval.rlogm[["dfbeta"]] >= 1, 1, any))
names(sospechosos4) <- NULL
cat("- Residuos con DFBeta mayor que 1: ")
print(sospechosos4)

# Finalmente, los casos no deberían desviarse significativamente
# de los límites recomendados para la razón de covarianza:
# CVRi > 1 + [3(k + 1)/n]
# CVRi < 1 – [3(k + 1)/n]
CVRi.lower <- 1 - 3 * apalancamiento.promedio
CVRi.upper <- 1 + 3 * apalancamiento.promedio
sospechosos5 <- which(eval.rlogm[["covariance.ratios"]] < CVRi.lower |
                        eval.rlogm[["covariance.ratios"]] > CVRi.upper)
cat("- Residuos con razón de covarianza fuera de rango ([", CVRi.lower, ", ",
    CVRi.upper, "]): ", sep = "")

print(sospechosos5)

sospechosos <- c(sospechosos1, sospechosos2, sospechosos3, sospechosos4,
                 sospechosos5)

sospechosos <- sort(unique(sospechosos))
cat("\nResumen de observaciones sospechosas:\n")

print(round(eval.rlogm[sospechosos,
                       c("cooks.distance", "leverage", "covariance.ratios")],
            3))

# Vemos que, si bien hay muchos residuos con valores atípicas, ninguno muestra
# un valor preocupante en la distancia de Cook o de apalancamiento. Por lo que
# el modelo no parece sufrir de casos muy influyentes.

# Ahora verifiquemos el supuesto de linealidad de la relación
xm1 <- data.frame(Logit = log(fitted(rlogs)/(1-fitted(rlogs))),
                   Weight = muestra.train[["Weight"]],
                   Hip.Girth = muestra.train[["Hip.Girth"]],
                   Bicep.Girth = muestra.train[["Bicep.Girth"]])
xm1.l <- pivot_longer(xm1, -Logit, names_to = "Predictor", values_to = "Valor")
pxm1 <- ggscatter(data = xm1.l, x = "Logit", y = "Valor", conf.int = TRUE) +
  geom_smooth(method = "loess") + 
  theme_bw() +
  facet_wrap(~ Predictor, scales = "free_y")
print(pxm1)

# Podemos ver que obviamente peso tiene una relación lineal perfecta (recordar
# que la usamos para definir la variable de salida EN) y las otras muestran un
# comportamiento mayoritariamente lineal con algunos valores atípicos.

# Revisemos ahora los residuos y si cumplen las condiciones necesarias
xm2 <- data.frame(Indice = 1:nrow(muestra.train),
                  Residuo.estandarizado = rstandard(rlogm))
pxm2 <- ggscatter(data = xm2, x = "Indice", y = "Residuo.estandarizado")
print(pxm2)

# Vemos que, como en el modelo simple, no aparece un patrón claro ni una 
# variación consistente de la varianza, lo que podemos confirmar con pruebas
# auxiliares:
print(shapiro.test(resid(rlogm)))
print(bptest(rlogm))

# Cumple bien con estos requisitos.

# Finalmente, revisamos que los residuos sean independientes
cat("\nPrueba de Durbin y Watson:\n")
print(durbinWatsonTest(rlogm))

# Al igual que con el modelo simple, no hay motivos para rechazar la idea de que
# los residuos son independientes.

# Concluimos que el modelo de RLogM que hemos obtenido tiene un buen ajuste y
# cumple de forma satisfactoria con las condiciones para ser considerado confiable.


################################################################################
# 8. Usando código estándar, evaluar el poder predictivo de los modelos con los
#    datos de las 30 personas que no se incluyeron en su construcción en
#    términos de sensibilidad y especificidad.
################################################################################

# Usaremos el umbral por defecto
umbral <- 0.5

# Primero calculemos el poder predictivo del modelo RLogS en los datos de 
# entrenamiento.
probs.trs <- fitted(rlogs)
preds.trs <- sapply(probs.trs,
                    function (p) ifelse (p >= umbral, "sobrepeso", "no sobrepeso"))
preds.trs <- factor(preds.trs, levels = levels(muestra.train[["EN"]]))
TP.trs <- sum(muestra.train[["EN"]] == "sobrepeso" & preds.trs == "sobrepeso")
FP.trs <- sum(muestra.train[["EN"]] == "no sobrepeso" & preds.trs == "sobrepeso")
TN.trs <- sum(muestra.train[["EN"]] == "no sobrepeso" & preds.trs == "no sobrepeso")
FN.trs <- sum(muestra.train[["EN"]] == "sobrepeso" & preds.trs == "no sobrepeso")
acc.trs <- (TP.trs + TN.trs) / (TP.trs + FP.trs + TN.trs + FN.trs)
sen.trs <- TP.trs / (TP.trs + FN.trs)
esp.trs <- TN.trs / (TN.trs + FP.trs)

# Ahora calculemos el poder predictivo del modelo RLogS en los datos de prueba 
probs.tes <- predict(rlogs, muestra.test, type = "response")
preds.tes <- sapply(probs.tes,
                    function (p) ifelse (p >= umbral, "sobrepeso", "no sobrepeso"))
preds.tes <- factor(preds.tes, levels = levels(muestra.test[["EN"]]))
TP.tes <- sum(muestra.test[["EN"]] == "sobrepeso" & preds.tes == "sobrepeso")
FP.tes <- sum(muestra.test[["EN"]] == "no sobrepeso" & preds.tes == "sobrepeso")
TN.tes <- sum(muestra.test[["EN"]] == "no sobrepeso" & preds.tes == "no sobrepeso")
FN.tes <- sum(muestra.test[["EN"]] == "sobrepeso" & preds.tes == "no sobrepeso")
acc.tes <- (TP.tes + TN.tes) / (TP.tes + FP.tes + TN.tes + FN.tes)
sen.tes <- TP.tes / (TP.tes + FN.tes)
esp.tes <- TN.tes / (TN.tes + FP.tes)

cat("\nRendimiento del modelo de RLogS:\n")
cat("    Exactitud entrenamiento:", sprintf("%.2f", acc.trs * 100), "\n")
cat("           Exactitud prueba:", sprintf("%.2f", acc.tes * 100), "\n")
cat(" Sensibilidad entrenamiento:", sprintf("%.2f", sen.trs * 100), "\n")
cat("        Sensibilidad prueba:", sprintf("%.2f", sen.tes * 100), "\n")
cat("Especificidad entrenamiento:", sprintf("%.2f", esp.trs * 100), "\n")
cat("       Especificidad prueba:", sprintf("%.2f", esp.tes * 100), "\n")
cat("\n")

# Podemos observar valores similares en el poder predictivo del modelo de
# RLogS en el conjunto de entrenamiento y en el conjunto de prueba. Esto
# es un indicador de que el modelo logra un buen nivel de aprendizaje. Así,
# debemos concluir que este modelo también es generalizable. 


# Ahora calculemos el poder predictivo del modelo RLogM en los datos de 
# entrenamiento.
probs.trm <- fitted(rlogm)
preds.trm <- sapply(probs.trm,
                    function (p) ifelse (p >= umbral, "sobrepeso", "no sobrepeso"))
preds.trm <- factor(preds.trm, levels = levels(muestra.train[["EN"]]))
TP.trm <- sum(muestra.train[["EN"]] == "sobrepeso" & preds.trm == "sobrepeso")
FP.trm <- sum(muestra.train[["EN"]] == "no sobrepeso" & preds.trm == "sobrepeso")
TN.trm <- sum(muestra.train[["EN"]] == "no sobrepeso" & preds.trm == "no sobrepeso")
FN.trm <- sum(muestra.train[["EN"]] == "sobrepeso" & preds.trm == "no sobrepeso")
acc.trm <- (TP.trm + TN.trm) / (TP.trm + FP.trm + TN.trm + FN.trm)
sen.trm <- TP.trm / (TP.trm + FN.trm)
esp.trm <- TN.trm / (TN.trm + FP.trm)

# Ahora calculemos el poder predictivo del modelo RLogS en los datos de prueba 
probs.tem <- predict(rlogm, muestra.test, type = "response")
preds.tem <- sapply(probs.tem,
                    function (p) ifelse (p >= umbral, "sobrepeso", "no sobrepeso"))
preds.tem <- factor(preds.tem, levels = levels(muestra.test[["EN"]]))
TP.tem <- sum(muestra.test[["EN"]] == "sobrepeso" & preds.tem == "sobrepeso")
FP.tem <- sum(muestra.test[["EN"]] == "no sobrepeso" & preds.tem == "sobrepeso")
TN.tem <- sum(muestra.test[["EN"]] == "no sobrepeso" & preds.tem == "no sobrepeso")
FN.tem <- sum(muestra.test[["EN"]] == "sobrepeso" & preds.tem == "no sobrepeso")
acc.tem <- (TP.tem + TN.tem) / (TP.tem + FP.tem + TN.tem + FN.tem)
sen.tem <- TP.tem / (TP.tem + FN.tem)
esp.tem <- TN.tem / (TN.tem + FP.tem)

cat("\nRendimiento del modelo de RLogM:\n")
cat("    Exactitud entrenamiento:", sprintf("%.2f", acc.trm * 100), "\n")
cat("           Exactitud prueba:", sprintf("%.2f", acc.tem * 100), "\n")
cat(" Sensibilidad entrenamiento:", sprintf("%.2f", sen.trm * 100), "\n")
cat("        Sensibilidad prueba:", sprintf("%.2f", sen.tem * 100), "\n")
cat("Especificidad entrenamiento:", sprintf("%.2f", esp.trm * 100), "\n")
cat("       Especificidad prueba:", sprintf("%.2f", esp.tem * 100), "\n")
cat("\n")

# Como en el caso del modelo simple, vemos que el poder predictivo del modelo de
# RLogM es similar en el conjunto de entrenamiento y en el conjunto de prueba,
# incluso parece un poco mejor en este último (aunque esto requiere de una prueba
# de McNemar para confirmarse). De esta forma, podemos concluir que este modelo
# también parece generalizable.

