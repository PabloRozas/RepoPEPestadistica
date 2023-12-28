
library(car)
library(caret)
library(ggfortify)
library(ggpubr)
library(leaps)
library(lmtest)
library(pROC)
library(tidyverse)

################################################################################
# 1. Definir la semilla a utilizar, que corresponde a los primeros cinco
#    dígitos del RUN del integrante de mayor edad del equipo.
# 2. Seleccionar una muestra de 100 personas, asegurando que la mitad tenga
#    estado nutricional “sobrepeso” y la otra mitad “no sobrepeso”.
################################################################################

# Fijamos la carpeta de trabajo
setwd("~/Downloads")

# Cargamos los datos y agregamos las variables IMC y EN
datos <- read.csv2("Datos Heinz et al 2003.csv")
datos.ext <- datos %>% mutate(IMC = Weight / (Height / 100)**2) %>%
  mutate(EN = ifelse(IMC < 25, "no sobrepeso", "sobrepeso"))
datos.ext[["Gender"]] <- factor(datos.ext[["Gender"]])
datos.ext[["EN"]] <- factor(datos.ext[["EN"]])

# Fijamos la semilla
set.seed(21111)

# Obtenemos la muestra
muestra.a <- datos.ext %>% filter(EN == "no sobrepeso") %>%
  sample_n(50, replace = FALSE)
muestra.b <- datos.ext %>% filter(EN == "sobrepeso") %>%
  sample_n(50, replace = FALSE)
muestra.ordenada <- rbind(muestra.a, muestra.b)

# Vamos a desordenar la muestra para que no queden ordenados los grupos
muestra <- muestra.ordenada[sample(1:100), ]

# Borramos las versiones que ya no usaremos (liberamos memoria)
rm(muestra.a, muestra.b, muestra.ordenada)

################################################################################
# 3. Usando las herramientas del paquete leaps, realizar una búsqueda exhaustiva
#    para seleccionar entre dos y ocho predictores que ayuden a estimar la
#    variable Peso (Weight), obviamente sin considerar las nuevas variables IMC
#    ni EN, y luego utilizar las funciones del paquete caret para construir un
#    modelo de regresión lineal múltiple con los predictores escogidos y
#    evaluarlo usando bootstrapping.
################################################################################

cat("\n")
cat("########################################################################\n")
cat("# Regresión lineal múltiple para la variable peso.\n")
cat("########################################################################\n")
cat("\n\n")

# Identificamos las columnas inútiles
IMC.i <- which(colnames(muestra) == "IMC")
EN.i <- which(colnames(muestra) == "EN")

# Seleccionamos predictores usando el método de todos los subconjuntos, teniendo
# cuidado de no considerar las variables inútiles.
sets <- regsubsets(Weight ~ ., data = muestra, nbest = 1, nvmax = 8, 
                   force.out = c(IMC.i, EN.i) - 1, method = "exhaustive")
sets.summ <- summary(sets)
mejor.i <- which.min(sets.summ[["bic"]])
pred.noms <- names(which(sets.summ[["which"]][mejor.i, ])[-1])
plot(sets)
cat("Mejores predictores:\n")
print(pred.noms)
cat("\n")

# Como todas las variables del mejor subconjunto son numéricas (y no necesitan
# variables indicadora), los nombres coinciden con los existentes en los datos,
# por lo que podemos usarlos directamente para construir el modelo.
peso.fmla.tex <- paste("Weight", paste(pred.noms, collapse = "+"), sep ="~")

# Vamos a ajustar modelo usando bootstrapping con B remuestreas
B = 2999
peso.rlm.tr <- train(formula(peso.fmla.tex), data = muestra,
                  method = "lm",
                  trControl = trainControl(method = "boot", number = B))
peso.rlm <- peso.rlm.tr[["finalModel"]]

cat("Modelo con estos predictores:\n")
print(summary(peso.rlm))
cat("\n")

# Cuando los modelos tienen tantos predictores, la probabilidad de que exista
# multicolinealidad aumenta. Por eso, es bueno que descartemos este potencial
# problema primero.
cat("Factores de inflación de la varianza:\n")
print(vif(peso.rlm))
cat("\n")
cat("Estadísticos de tolerancia:\n")
print(1 / vif(peso.rlm))
cat("\n")

# Como supusimos, hay indicios de multicolinealidad. Eliminemos un predictor.
# ¿Pero cuál? Los que tienen mayor factor de inflación de la varianza son 
# Chest.Girth y Forearm.Girth, los que probablemente están correlacionados.
# Mirando al modelo, vemos que efectivamente Forearm.Girth aporta poco al
# modelo, que ya tiene Chest.Girth. Problemos eliminando esta variable.
Forearm.Girth.i <- which(pred.noms == "Forearm.Girth")
pred.noms <- pred.noms[-Forearm.Girth.i]
peso.fmla.tex <- paste("Weight", paste(pred.noms, collapse = "+"), sep ="~")
peso.rlm.tr <- train(formula(peso.fmla.tex), data = muestra,
                  method = "lm",
                  trControl = trainControl(method = "boot", number = B))
peso.rlm <- peso.rlm.tr[["finalModel"]]

cat("Modelo mejorado (1):\n")
print(summary(peso.rlm))
cat("\n")

cat("Factores de inflación de la varianza del modelo mejorado (1):\n")
print(vif(peso.rlm))
cat("\n")
cat("Estadísticos de tolerancia del modelo mejorado (1):\n")
print(1 / vif(peso.rlm))
cat("\n")

# Sigue habiendo un par de predictores con VIF alto. Eliminemos Chest.Girth,
# puesto que el grosor de la cintura podría ayudar más a predecir el peso.
Chest.Girth.i <- which(pred.noms == "Chest.Girth")
pred.noms <- pred.noms[-Chest.Girth.i]
peso.fmla.tex <- paste("Weight", paste(pred.noms, collapse = "+"), sep ="~")
peso.rlm.tr <- train(formula(peso.fmla.tex), data = muestra,
                  method = "lm",
                  trControl = trainControl(method = "boot", number = B))
peso.rlm <- peso.rlm.tr[["finalModel"]]

cat("Modelo mejorado (2):\n")
print(summary(peso.rlm))
cat("\n")

cat("Factores de inflación de la varianza del modelo mejorado (2):\n")
print(vif(peso.rlm))
cat("\n")
cat("Estadísticos de tolerancia del modelo mejorado (2):\n")
print(1 / vif(peso.rlm))
cat("\n")

# Ahora conseguimos un modelo con mejores factores de inflación de la varianza

# El modelo obtenido presenta un R^2 ajustado de 0,9466. Esto significa que el
# modelo obtenido explica el 94,66% de la variabilidad de los datos, por lo que
# consigue un muy buen ajuste.

# Veamos la calidad predictiva del modelo.
cat("Rendimiento del modelo mejorado (2):\n")
print(peso.rlm.tr[["results"]])
cat("\n")

# Vemos que el error promedio que el modelo comete en sus estimaciones es de
# 3,29 Kg, lo que no es mucho si se considera que la variable de salida varía
# (aproximadamente) entre 46 y 105 Kg. Podemos revisar un histograma de estos
# errores para verificar su dispersión.
peso.p1.datos <- data.frame(RMSE = peso.rlm.tr[["resample"]][["RMSE"]])
peso.p1 <- gghistogram(peso.p1.datos, x = "RMSE", bins = 30)
print(peso.p1)

# En este histograma vemos un comportamiento normal del error, por lo que el
# modelo muestra estabilidad en las diferentes remuestras, indicador de su
# generalidad.

# Solo falta mirar su confiabilidad

# Comencemos por el supuesto de linealidad de la relación
peso.p2.datos.ancho <- muestra %>% select(all_of(c("Weight", pred.noms)))
peso.p2.datos <- pivot_longer(peso.p2.datos.ancho, -Weight,
                              names_to = "Predictor", values_to = "Valor")
peso.p2 <- ggscatter(data = peso.p2.datos, x = "Weight", y = "Valor",
                     conf.int = TRUE) +
  geom_smooth(method = "loess", span = 5) + 
  theme_gray() +
  facet_wrap(~ Predictor, scales = "free_y")
print(peso.p2)

# Vemos que las relaciones son razonablemente lineales, con la incertidumbre
# usual en los extremos. Si quisiéramos ser muy exigentes, tal vez deberíamos
# sacar la variable edad del modelo, pues parece tener poca relevancia y mayor
# incertidumbre. Se deja como ejercicio.

# Revisemos ahora los residuos y si cumplen las condiciones necesarias
peso.p3 <- autoplot(peso.rlm, which = 1:3, ncol = 3,
                    label.size = 3, label.n = 5, label.colour = 'blue',)
print(peso.p3)

# Vemos que el gráfico 1 (Residual vs Fitted) no aparece un patrón preocupante,
# los extremos no son muy rectos, pero eso puede deberse a que hay pocos datos
# en esas zonas. El gráfico 2 (Normal Q-Q) refleja la misma información, con desviación leves
# en los extremos.Por último, la varianza parece bastante constante en gráfico 3.
# Podemos confirmar con pruebas auxiliares si tenemos dudas:
print(shapiro.test(resid(peso.rlm)))
print(bptest(peso.rlm))
cat("\n")

# Así, los residuos cumplen bien con los requisitos de normalidad y homocedas-
# ticidad.

# Luego, revisamos que los residuos son independientes
cat("Prueba de Durbin y Watson del modelo mejorado (2):\n")
print(durbinWatsonTest(peso.rlm))
cat("\n")

# Sin problemas en este aspecto.

# Solo queda revisar que no existan casos demasiado influyentes.
peso.p4 <- autoplot(peso.rlm, which = c(5, 4), ncol = 2,
                    label.size = 3, label.n = 5, label.colour = 'blue',)
print(peso.p4)

# Vemos que, si bien hay muchos residuos con valores atípicos, solo unos pocos
# presentan valores altos de apalancamiento (en este caso: 2(k+1)/100 = 0,14) y
# solo uno parece tener un valor muy alto (3(k+1)/100 = 0,21). Sin embargo,
# todos exhiben una distancia de Cook muy baja (<0,1), por lo que el modelo no
# parece sufrir de casos muy influyentes.

# Con todas estas verificaciones, podemos concluir que el modelo mejorado (2)
# es confiable y generalizable, aunque con un error algo alto (> 3 Kg).



################################################################################
# 4. Haciendo un poco de investigación sobre el paquete caret, en particular
#    cómo hacer Recursive Feature Elimination (RFE), construir un modelo de
#    regresión lineal múltiple para predecir la variable IMC que incluya entre
#    10 y 20 predictores, seleccionando el conjunto de variables que maximice R2
#    y que use cinco repeticiones de validación cruzada de cinco pliegues para
#    evitar el sobreajuste (obviamente no se debe considerar las variables Peso,
#    Estatura ni estado nutricional –Weight, Height, EN respectivamente). 
################################################################################

cat("\n")
cat("########################################################################\n")
cat("# Regresión lineal múltiple para la variable IMC.\n")
cat("########################################################################\n")
cat("\n\n")

# Quitamos las columnas inútiles de los datos
imc.muestra <- muestra %>% select(-Weight, -Height, -EN)

# Caret implementa la regresión escalonada hacia atrás (bajo el nombre de
# Recursive Feature Elimination) mediante la función rfe().
# Se pueden definir alternativas de control que guíen la búsqueda, incluyendo
# funciones wrapper para el tipo de modelo. El paquete caret proporciona la
# función wrapper lmFuncs para modelos de regresión lineal.

imc.control <- rfeControl(functions = lmFuncs, method = "repeatedcv",
                          number = 5, repeats = 5, verbose = FALSE)

imc.rfe <- rfe(IMC ~ ., data = imc.muestra, rfeControl = imc.control,
               sizes = 10:20, metric = "Rsquared")
imc.rlm <- imc.rfe[["fit"]]

cat("Modelo obtenido con RFE:\n")
print(summary(imc.rlm))
cat("\n")

# La búsqueda obtuvo un modelo que considera 16 predictores. Sin embargo,
# al observar el detalle, vemos que varios parecen ser irrelevantes.
# La probabilidad de que exista multicolinealidad es alta, por lo que es bueno
# que revisemos este aspecto primero.
cat("Factores de inflación de la varianza:\n")
print(vif(imc.rlm))
cat("\n")
cat("Estadísticos de tolerancia:\n")
print(1 / vif(imc.rlm))
cat("\n")

# Uff! Vemos que Bicep.Girth, Wrist.Minimum.Girth, Gender y Elbows.diameter
# exhiben valores de VIF muy altos y no aportan significativamente al modelo.
# Quitemos estos predictores primero.

imc.rlm.tr <- train(IMC ~ Forearm.Girth + Knees.diameter + Ankles.diameter +
                      Wrists.diameter + Thigh.Girth + Calf.Maximum.Girth +
                      Biacromial.diameter + Ankle.Minimum.Girth + Knee.Girth +
                      Waist.Girth + Bitrochanteric.diameter + Chest.depth,
                    data = imc.muestra,
                    method = "lm",
                    trControl = trainControl(method = "repeatedcv",
                                             number = 5, repeats = 5))
imc.rlm <- imc.rlm.tr[["finalModel"]]

cat("Modelo mejorado (1):\n")
print(summary(imc.rlm))
cat("\n")

cat("Factores de inflación de la varianza del modelo mejorado (1):\n")
print(vif(imc.rlm))
cat("\n")
cat("Estadísticos de tolerancia del modelo mejorado (1):\n")
print(1 / vif(imc.rlm))
cat("\n")

# Todavía hay predictores con altos valores de VIF. Recordando que se nos
# pide un modelo con al menos 10 predictores, solo podemos eliminar otras
# otras dos variables. Veamos una gráfica de las correlaciones entre estos
# predictores.
imc.pred.noms <- imc.rlm[["xNames"]]
imc.pred <- as.matrix(imc.muestra[, imc.pred.noms])
imc.pred.cormat <- abs(cor(imc.pred))
print(autoplot(imc.pred.cormat))

# Vemos que Ankle.Minimum.Girth tiene altos valores de correlación con la
# mayoría de las otras variables y no aparece como relevante para el modelo,
# porque lo que es el primer predictor seleccionado para desaparecer.
# Otra variable que muestra altas correlaciones es Forearm.Girth, pero
# esta es estimada como relevante para el modelo. Tratemos de eliminar algo
# de la colinealidad asociada a este predictor eliminando la variable
# Wrists.diameter, que tiene alta correlación con Forearm.Girth y no
# aporta significativamente al modelo.
imc.rlm.tr <- train(IMC ~ Forearm.Girth + Knees.diameter + Ankles.diameter +
                      Thigh.Girth + Calf.Maximum.Girth + Biacromial.diameter +
                      Knee.Girth + Waist.Girth + Bitrochanteric.diameter +
                      Chest.depth,
                    data = imc.muestra,
                    method = "lm",
                    trControl = trainControl(method = "repeatedcv",
                                             number = 5, repeats = 5))
imc.rlm <- imc.rlm.tr[["finalModel"]]

cat("Modelo mejorado (2):\n")
print(summary(imc.rlm))
cat("\n")

cat("Factores de inflación de la varianza del modelo mejorado (2):\n")
print(vif(imc.rlm))
cat("\n")
cat("Estadísticos de tolerancia del modelo mejorado (2):\n")
print(1 / vif(imc.rlm))
cat("\n")

# Vemos que después de estas iteraciones, existen predictores con valores de
# VIF y tolerancia *apenas* aceptables... sería prudente seguir quitando estos
# predictores hasta conseguir un conjunto de variables que no muestre señales
# de multicolinealidad, pero eso nos dejaría un modelo con menos de 10 predic-
# tores y no cumplíamos lo solicitado en el enunciado. Solo por esta razón,
# nos vamos a detener aquí, declarando al modelo como *relativamente* confiable.

# Veamos la calidad predictiva del modelo.
cat("\n\n")
cat("Rendimiento del modelo mejorado (2):\n")
print(imc.rlm.tr[["results"]])
cat("\n")

# Vemos que el error promedio que el modelo comete en sus estimaciones es de
# 1,26 kg m^(-2), lo que parece razonable considerando que la variable de salida
# varía (aproximadamente) entre 18 y 35 kg m^(-2). Revisemos el histograma de
# estos errores para verificar su dispersión.
imc.p1.datos <- data.frame(RMSE = imc.rlm.tr[["resample"]][["RMSE"]])
imc.p1 <- gghistogram(imc.p1.datos, x = "RMSE", bins = 5)
print(imc.p1)

# Al menos no parece que el error haya saltado de un valor a otro entre los
# pliegues evaluados. Podemos decir que parece haber alcanzado un cierto nivel
# de generalidad.

# Veamos otros aspectos de la confiabilidad (a parte de la multicolinealidad)

# Comencemos por el supuesto de linealidad
imc.pred.noms <- colnames(imc.rlm$model)[-1]
imc.p2.datos.ancho <- imc.muestra %>% select(all_of(c("IMC", imc.pred.noms)))
imc.p2.datos <- pivot_longer(imc.p2.datos.ancho, -IMC,
                              names_to = "Predictor", values_to = "Valor")
imc.p2 <- ggscatter(data = imc.p2.datos, x = "IMC", y = "Valor",
                     conf.int = TRUE) +
  geom_smooth(method = "loess", span = 5) + 
  theme_gray() +
  facet_wrap(~ Predictor, scales = "free_y")
print(imc.p2)

# Vemos que las relaciones son razonablemente lineales en la mayoría de
# los casos, si bien se observan desvíos en casos con IMC alto, aunque
# esto podría deberse a que hay muy pocos ejemplos de estas personas en
# la muestra de datos.
# Más preocupantes son los casos de los predictores Ankles.diameter y
# Bitrochanteric.diameter, en que la relación lineal es más cuestionable.
# Si pudiéramos seguir eliminando predictores, estos dos serían buenos
# candidatos.

# Revisemos los residuos
imc.p3 <- autoplot(imc.rlm, which = 1:3, ncol = 3,
                    label.size = 3, label.n = 5, label.colour = 'blue',)
print(imc.p3)

# Vemos, nuevamente, que hay un caso con un error atípico preocupante y otro que
# está muy alejado del resto de la nube de datos con un IMC muy alto. Este último
# no parece influir tanto en la normalidad de los residuos ni en el modelo pero,
# si bien no aparece algún patrón preocupante, es probable que el primero afecte
# el resultado de las pruebas auxiliares:
print(shapiro.test(resid(imc.rlm)))
print(bptest(imc.rlm))
cat("\n")

# No hay problemas graves con la normalidad y homocedasticidad de los residuos.

# Revisemos si existen casos influyentes.
imc.p4 <- autoplot(imc.rlm, which = c(5, 4), ncol = 2,
                    label.size = 3, label.n = 5, label.colour = 'blue',)
print(imc.p4)

# Vemos que, si bien hay varios residuos que podrían considerarse con valor
# atípico, ninguno influye demasiado en el modelo. Hay dos casos que tienen
# valores altos de apalancamiento (en este caso: 2(k+1)/100 = 0,2) y solo
# uno de ellos tiene un valor muy alto (3(k+1)/100 = 0,3). Aunque podríamos
# hacer nada al respecto, ya que ambos presentan una distancia de Cook baja
# (<0,25), vamos a eliminemos estos dos casos como ejercicio académico.

# Primero identificamos los casos a eliminar
imc.res <-resid(imc.rlm)
imn.res.noms <-names(imc.res)
i.lev <- which(hatvalues(imc.rlm) > 0.2)

# Los quitamos de la muestra y aprovechamos de desordenarla
imc.muestra2 <- imc.muestra[-i.lev, ]
imc.muestra2 <- imc.muestra2[sample(1:nrow(imc.muestra2)), ]

# Obtenemos el modelo mejorado
imc.rlm.tr <- train(IMC ~ Forearm.Girth + Knees.diameter + Ankles.diameter +
                      Thigh.Girth + Calf.Maximum.Girth + Biacromial.diameter +
                      Knee.Girth + Waist.Girth + Bitrochanteric.diameter +
                      Chest.depth,
                    data = imc.muestra2,
                    method = "lm",
                    trControl = trainControl(method = "repeatedcv",
                                             number = 5, repeats = 5))
imc.rlm <- imc.rlm.tr[["finalModel"]]

# Revisemos nuevamente si existen casos influyentes.
imc.p5 <- autoplot(imc.rlm, which = c(5, 4), ncol = 2,
                   label.size = 3, label.n = 5, label.colour = 'blue',)
print(imc.p5)

# Vemos que hay algunos casos con valores de apalancamiento en el borde de
# lo aceptable (en este caso: 2(k+1)/98 = 0,204).
# Como el modelo fue cambiado, deberíamos revisar nuevamente las condiciones
# anteriores, pero eso queda como ejercicio. Es probable que no haya mucha
# diferencia por eliminar dos casos un poco influyentes.

# Finalmente, revisamos que los residuos son independientes (fijando una
# semilla para obtener resultados consistentes).
set.seed(21111 * 1421)
cat("Prueba de Durbin y Watson del modelo mejorado (3):\n")
print(durbinWatsonTest(imc.rlm))
cat("\n")

# Vemos que no hay indicios de que existan dependencias entre los residuos

# Así, podemos concluir que el modelo mejorado (3) es relativamente confiable
# (hay signos de multicolinealidad, los que no se pueden mejorar sin bajar de
# 10 predictores) y generalizable, con un error aceptable.


################################################################################
# 5. Usando RFE, construir un modelo de regresión logística múltiple para la
#    variable EN que incluya el conjunto de entre dos y seis predictores que
#    entregue la mejor curva ROC y que utilice validación cruzada dejando uno
#    fuera para evitar el sobreajuste (obviamente no se debe considerar las
#    variables Peso, Estatura –Weight y Height respectivamente– ni IMC).
################################################################################

cat("\n")
cat("########################################################################\n")
cat("# Regresión logística múltiple para la variable EN.\n")
cat("########################################################################\n")
cat("\n\n")

# Quitamos las columnas inútiles de los datos y generamos etiquetas que sean nombres
# de variables válidos para R (sino lo son, train() puede dar errores).
en.muestra <- muestra %>% select(-Weight, -Height, -IMC) %>%
  mutate(EN = factor(EN, labels = make.names(levels(EN))))

# Definimos las condiciones de búsqueda de los predictores utilizando la función
# wrapper lrFuncs para modelos de regresión logística, haciendo unos ajustes
# para que use el área bajo la curva ROC como métrica de búsqueda (que parece
# ser accuracy por defecto).

lrFuncs[["summary"]] <- twoClassSummary
en.rfe.control <- rfeControl(functions = lrFuncs, method = "LOOCV", number = 1,
                             verbose = FALSE)
en.tr.control <- trainControl(method = "none", classProbs = TRUE,
                              summaryFunction = twoClassSummary)
en.rfe <- rfe(EN ~ ., data = en.muestra, sizes = 2:6, rfeControl = en.rfe.control,
              metric = "ROC", trControl = en.tr.control)
en.rlog <- en.rfe[["fit"]]

cat("Modelo obtenido con RFE:\n")
print(summary(en.rlog))
cat("\n")

# Si bien durante la búsqueda hubo modelos para los cuales el cálculo de sus
# coeficientes no pudo converger, se pudo obtener uno con cinco predictores,
# uno de los cuales parece poco relevante. Revisemos si es que hay señales
# de que exista multicolinealidad.
cat("Factores de inflación de la varianza:\n")
print(vif(en.rlog))
cat("\n")
cat("Estadísticos de tolerancia:\n")
print(1 / vif(en.rlog))
cat("\n")

# Vemos que Chest.diameter, que no aparece como relevante en el modelo, tiene 
# un valor de VIF relativamente alto. Quitemos esta variable del modelo.
en.tr.control <- trainControl(method = "LOOCV", classProbs = TRUE,
                              summaryFunction = twoClassSummary)
en.rlog.tr <- train(EN ~ Waist.Girth + Forearm.Girth + Thigh.Girth + 
                      Biacromial.diameter,
                    data = en.muestra, method = "glm",
                    metric = "ROC", trControl = en.tr.control)
en.rlog <- en.rlog.tr[["finalModel"]]

cat("Modelo mejorado (1):\n")
print(summary(en.rlog))
cat("\n")

# Revisemos como quedaron los factores de inflación de la varianza.
cat("Factores de inflación de la varianza del modelo mejorado (1):\n")
print(vif(en.rlog))
cat("\n")
cat("Estadísticos de tolerancia del modelo mejorado (1):\n")
print(1 / vif(en.rlog))
cat("\n")

# ¡Bien! El modelo consigue una importante reducción de "devianza", lo
# que nos indica que el modelo consigue un buen ajuste. Además, parece
# no haber indicios de multicolinealidad.

# Veamos la calidad predictiva del modelo.
cat("Rendimiento del modelo mejorado (1):\n")
print(en.rlog.tr[["results"]])
cat("\n")

# Vemos que tiene muy buen rendimiento, con un área bajo la curva ROC de
# 0.93 (Sens = 0.88, Spec = 0.92).

# Revisemos el supuesto de linealidad. Recordemos, que en caso de la regresión
# logística debe darse con el logaritmo de los odds (es decir, el logit() de la
# probabilidad estimada.
en.pred.noms <- en.rlog[["xNames"]]
en.p1.datos.anchos <- en.muestra %>% select(all_of(en.pred.noms)) %>%
  mutate(Logit = logit(fitted(en.rlog)))
en.p1.datos <- pivot_longer(en.p1.datos.anchos, -Logit,
                             names_to = "Predictor", values_to = "Valor")
en.p1 <- ggscatter(data = en.p1.datos, x = "Logit", y = "Valor",
                    conf.int = TRUE) +
  geom_smooth(method = "loess", span = 5) + 
  theme_gray() +
  facet_wrap(~ Predictor, scales = "free_y")
print(en.p1)

# Vemos que las relaciones son razonablemente lineales con la típica
# desviación en el extremo donde hay pocos casos observados.
# Puede que el comportamiento de Bitrochanteric.diameter sea más
# preocupante, y probablemente sería bueno comparar este modelo con
# uno sin esta variable. Eso se deja como *ejercicio*.

# Revisemos los residuos
en.p2 <- autoplot(en.rlog, which = 1:3, ncol = 3,
                  label.size = 3, label.n = 5, label.colour = 'blue',)
print(en.p2)

# Vemos que no es simple evaluar la normalidad de los residuos con estos
# gráficos. El 2do sí permite apreciar que parece haber desviaciones
# importantes de normalidad. Es posible que eliminando el predictor
# Bitrochanteric.diameter, que mostró una linealidad algo cuestionable,
# se obtengan residuos que se ajusten más a una distribución normal.
# Se agrega esta revisión al *ejercicio* que se dejó planteado.

# Vamos a cambiar el gráfico 1 y 3, a ver si hacemos más sentido de lo
# que se ve en ellos.

# Obtenemos los datos de estos gráficos y le agregamos unos índices para
# graficarlos sin considerar su probabilidad estimada.
en.p2.datos2 <- cbind(.index = 1:nrow(en.muestra), fortify(en.rlog),
                      .scale = sqrt(abs(rstandard(en.rlog))))
en.p2.1 <- ggscatter(en.p2.datos2, x = ".index", y = ".resid",
                     title = "Residuals", xlab = "Indice",
                     ylab = "Residuals") +
  geom_smooth(method = "loess", span = 5) +
  theme_gray()
en.p2.3 <- ggscatter(en.p2.datos2, x = ".index", y = ".scale",
                     title = "Scale-Location", xlab = "Indice",
                     ylab = "Root of Standardized Residuals") +
  geom_smooth(method = "loess", span = 5) +
  theme_gray()
en.p2.2 <- en.p2
en.p2.2[[1]] <- en.p2.1
en.p2.2[[3]] <- en.p2.3
print(en.p2.2)

# Estos gráficos se parecen más a lo que conocemos. Vemos que los residuos
# (sin orden) se ven aleatorios, aunque con varios casos atípicos, espe-
# cialmente bajo el cero. Mirando el gráfico de scale-location, se aprecia
# un patrón, no muy claro, que tal vez también se puede deber al efecto de
# la variable Bitrochanteric.diameter. Así es que revisar si se mantiene
# al desarrollar *ejercicio* que se dejó planteado.

# Revisemos los residuos si hay casos influyentes
en.p3 <- autoplot(en.rlog, which = 4, ncol = 1,
                  label.size = 3, label.n = 5, label.colour = 'blue',)
print(en.p3)

# Vemos que solo hay un caso que podría ser considerado como influyente en
# comparación con los otros, porque no excede 0,4 en distancia de Cook.
# Queda como ejercicio eliminar ese caso y evaluar el modelo mejorado que
# resulte.

# Finalmente, revisamos que los residuos son independientes (fijando una
# semilla para obtener resultados consistentes).
set.seed(21111 * 447)
cat("Prueba de Durbin y Watson del modelo mejorado (1):\n")
print(durbinWatsonTest(en.rlog))
cat("\n")

# No hay motivos para rechazar la idea de que no existe dependencia entre
# los residuos.

# Así, podemos concluir que el modelo mejorado (1) es relativamente
# confiable, pues queda en duda si existe una relación suficientemente
# lineal de la variable de salida EN y el predictor Bitrochanteric.diameter.
# Esto podría también explicar la falta de normalidad de los residuos y
# sería prudente intentar mejorar estos aspectos.
# Teniendo esto en consideración, el modelo obtenido muestra un buen nivel
# de ajuste y excelente capacidad clasificatoria estimada generalmente con
# el método LOOCV.


################################################################################
# 6. Pronunciarse sobre la confiabilidad y el poder predictivo de los modelos.
################################################################################

# Se hizo en cada uno de los modelos creados

