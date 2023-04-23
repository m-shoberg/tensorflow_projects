#add dependencies
import tensorflow as tf
import numpy as np

#prepare data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

#design model: 1 layer - single neron nural network
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
#compline model - loss and optimimizer | #stochastic gradient descent (sgd)
model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.MeanSquaredError())
#train neural network (similar results can be achieved with smaller number of epochs)
model.fit(xs, ys, epochs=500)

#make prediction for y when x=10 
print('Prediction: ', model.predict([10.0]))

