"""
Aprendizaje No-supervisado con Modelos Generativos Profundos

Fernando Arribas Jara

TUTOR: Daniel Hernández Lobato

UNIVERSIDAD AUTONOMA DE MADRID

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
#Importar MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#Variables de Train
num_epocas =20
batch_size = 100
learning_rate = 1e-3
n_datos = mnist.train.num_examples
num_batch = int(n_datos / batch_size)
#Variables del dataset
tam_imagen = 784 # Imagenes de 28x28 pixels
tam_latente = 2
tam_hidden_layer = 500

#Variables del model
x = tf.placeholder(tf.float32, shape=[None, tam_imagen]) #Entrada de datos, imagenes

# 1) Encoder Q(z|x, phi)

#Semilla aleatoria
tf.set_random_seed(0)
#Pesos y biases
W_encoder = tf.Variable(tf.random_normal([tam_imagen, tam_hidden_layer], stddev= tf.pow(float(tam_imagen), -0.5)))
b_encoder = tf.Variable(tf.random_normal([tam_hidden_layer], stddev= tf.pow(float(tam_hidden_layer), -0.5)))

W_z_var = tf.Variable(tf.random_normal([tam_hidden_layer,tam_latente], stddev=tf.pow(float(tam_hidden_layer), -0.5)))
b_z_var = tf.Variable(tf.random_normal([tam_latente], stddev=tf.pow(float(tam_latente), -0.5)))

W_z_mean = tf.Variable(tf.truncated_normal([tam_hidden_layer,tam_latente], stddev=tf.pow(float(tam_hidden_layer), -0.5)))
b_z_mean = tf.Variable(tf.truncated_normal([tam_latente], stddev=tf.pow(float(tam_latente), -0.5)))

#Model del Encoder
encoder = tf.matmul(x, W_encoder) + b_encoder
encoder = tf.nn.relu(encoder)


# 2) Crear muestras de z

#Mean
z_mean = tf.matmul(encoder,W_z_mean)+b_z_mean
#Std
z_var = tf.matmul(encoder, W_z_var)+b_z_var

epsilon = tf.random_normal(tf.shape(z_var)) #Parametros por defecto: mean = 0  std= 1
z = z_mean + (tf.multiply(tf.sqrt(tf.exp(z_var)),epsilon))

# 3) Decoder:  P(x|z, theta)

#Pesos y biases
W_decoder = tf.Variable(tf.random_normal([tam_latente, tam_hidden_layer], stddev=tf.pow(float(tam_latente), -0.5)))
b_decoder = tf.Variable(tf.random_normal([tam_hidden_layer], stddev= tf.pow(float(tam_hidden_layer), -0.5)))

W_decoder_out = tf.Variable(tf.random_normal([tam_hidden_layer, tam_imagen], stddev=tf.pow(float(tam_hidden_layer), -0.5)))
b_decoder_out = tf.Variable(tf.random_normal([tam_imagen], stddev= tf.pow(float(tam_imagen), -0.5)))

#Model del Decoder
decoder = tf.matmul(z, W_decoder) + b_decoder
decoder = tf.nn.relu(decoder)
decoder = tf.matmul(decoder, W_decoder_out) + b_decoder_out
decoder = tf.nn.sigmoid(decoder)


#Para evitar que que el calculo de gradientes de error cuando los logaritmos son log(x) con x~0 se suma un sesgo  1e-9
decoder = tf.clip_by_value(decoder, 1e-10, 1 - 1e-10)
likelihood = tf.reduce_mean(tf.reduce_sum(x * tf.log(decoder) + (1 - x) * tf.log(1 - decoder), 1))
#Divergencia KL:  -D_KL(q(z)||p(z))
KL = (1/2) * tf.reduce_sum(1 + z_var - tf.square(z_mean) - tf.exp(z_var), 1)

#El likelihood se escala al tam de los batches
ELBO =  tf.reduce_mean(KL + (likelihood*(n_datos/num_batch)))

#Como nuestro objetivo es maximizar el ELBO vamos en sentido opuesto  
function_coste = -ELBO

#Optimizador Adam
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-08)
train_step = optimizer.minimize(function_coste)

print('Entrenamiento')
session = tf.InteractiveSession()
tf.global_variables_initializer().run()
valores_coste = []
for epoca in range(1, num_epocas+1):   
    average_coste = 0
    inicio = time.time()
    for i in range(num_batch):         
        batch, _ = mnist.train.next_batch(batch_size)        
        _, coste = session.run([train_step, function_coste], feed_dict = {x: batch} )
        average_coste = (average_coste + coste) / num_batch
        valores_coste.append(coste)
    fin = time.time()
    tiempo = fin - inicio
    print('Epoca: %d Tiempo: %f  Loss= %f' % (epoca,tiempo,average_coste))




