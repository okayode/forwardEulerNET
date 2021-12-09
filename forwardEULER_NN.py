import numpy as np
import tensorflow.compat.v1 as tf
import timeit
 
tf.disable_v2_behavior()
np.random.seed(1234)
tf.set_random_seed(1234)

class f_Euler_NET:
     def __init__(self,dt,X,layers):
         self.dt = dt
         self.X = X

         self.S = X.shape[0]
         self.N = X.shape[1]
         self.D = X.shape[2]

         self.layers = layers

         # tf placeholders and graph
         self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
         self.X_tf = tf.placeholder(tf.float32, shape=[self.S, None, self.D])
         self.X_star_tf = tf.placeholder(tf.float32, shape=[None, self.D])

         scope_name = str(np.random.randint(1e6))
         with tf.variable_scope(scope_name) as scope:
              self.f_pred = self.neural_net(self.X_star_tf)
         with tf.variable_scope(scope, reuse=True):
              self.Y_pred = self.net_Y(self.X_tf)

         self.loss = self.D*tf.reduce_mean(tf.square(self.Y_pred))

         self.optimizer_Adam = tf.train.AdamOptimizer()
         self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

         init = tf.global_variables_initializer()
         self.sess.run(init)

     def neural_net(self, H):
         num_layers = len(self.layers)
         for l in range(0,num_layers-2):
             with tf.variable_scope("layer%d" %(l+1)):
                  H = tf.layers.dense(inputs=H, units=self.layers[l+1], activation=tf.nn.tanh)
         with tf.variable_scope("layer%d" %(num_layers-1)):
              H = tf.layers.dense(inputs=H, units=self.layers[-1], activation=None)
         return H

     def net_F(self, X):
         X_reshaped = tf.reshape(X, [-1,self.D])
         F_reshaped = self.neural_net(X_reshaped)
         F = tf.reshape(F_reshaped, [self.S,-1,self.D])
         return F

     def net_Y(self, X):
         M = 1
         Y = -X[:,M:,:] + X[:,M-1:-1,:] + self.dt*self.net_F(X[:,M-1:-1,:])
         return Y

     def train(self, N_Iter):
         tf_dict = {self.X_tf: self.X}
         start_time = timeit.default_timer()
         for it in range(N_Iter):
             self.sess.run(self.train_op_Adam, tf_dict)
             if it % 100 == 0:
                   elapsed = timeit.default_timer() - start_time
                   loss_value = self.sess.run(self.loss, tf_dict)
                   print('It: %d, Loss: %.3e, Time: %.2f' %
                          (it, loss_value, elapsed))
                   start_time = timeit.default_timer()

     def predict_f(self, X_star):
         F_star = self.sess.run(self.f_pred, {self.X_star_tf: X_star})
         return F_star
