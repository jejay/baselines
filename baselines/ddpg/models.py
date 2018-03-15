import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]

class WeightSharingActor(Model):
    def __init__(self, specification, name='weight-sharing-actor', layer_norm=True):
        super(WeightSharingActor, self).__init__(name=name)
        self.specification = specification
        self.specification = {
            'commander': {
                'hidden_layers': 1,
                'units': 64
            },
            'controllers': [
                {
                    'name': 'leg',
                    'hidden_layers': 0,
                    'units': 16,
                    'action_indice_groups': [[0, 1], [2, 3], [4, 5], [6, 7]]
                },
                #{
                #    'name': 'weird',
                #    'hidden_layers': 1,
                #    'units': 64,
                #    'action_indice_groups': [[1,2,3], [4,5,6]]
                #},
                #{
                #    'name': 'awkward',
                #    'hidden_layers': 1,
                #    'units': 64,
                #    'action_indice_groups': [[4]]
                #},
            ]
        }
        self.layer_norm = layer_norm
        self.indices = []
        for controller in self.specification['controllers']:
            controller['output_length'] = len(controller['action_indice_groups'][0])
            for action_indice_group in controller['action_indice_groups']:
                assert controller['output_length'] == len(action_indice_group), \
                    "Controller %r has an action_indice_group length mismatch. All groups should be the same length." % controller
                self.indices += action_indice_group
        for i in range(max(self.indices)):
            assert i in self.indices, \
                "Action index %r not found." % i
        self.nb_actions = max(self.indices) + 1

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            x = obs
            for i in range(self.specification['commander']['hidden_layers']):
                x = tf.layers.dense(x, self.specification['commander']['units'])
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                
            output = tf.zeros(shape=[1, self.nb_actions])

            for controller in self.specification['controllers']:
                for aig_idx, action_indice_group in enumerate(controller['action_indice_groups']):

                    with tf.variable_scope(controller['name']+'-branch-'+str(aig_idx)):
                        # This layer splits the controllers. Weights can not be shared here.
                        x_ = tf.layers.dense(x, controller['units'])
                        if self.layer_norm:
                            x_ = tc.layers.layer_norm(x_, center=True, scale=True)
                        x_ = tf.nn.relu(x_)

                    with tf.variable_scope(controller['name']) as controller_scope:
                        # Starting variable/weights sharing if we are in the second or higher action index group
                        if aig_idx > 0:
                            controller_scope.reuse_variables()

                        for i in range(controller['hidden_layers']):

                            # controllers hidden layer
                            x_ = tf.layers.dense(x_, controller['units'],
                                                 name='hidden-layer-'+str(i))
                            if self.layer_norm:
                                x_ = tc.layers.layer_norm(x_, center=True, scale=True)
                            x_ = tf.nn.relu(x_)

                        #controllers output layer
                        x_ = tf.layers.dense(x_, controller['output_length'],
                                             kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                                             name='final-layer')

                        output_projection = np.zeros((controller['output_length'], self.nb_actions))
                        for controller_output_index, action_index in enumerate(action_indice_group):
                            output_projection[controller_output_index, action_index] = 1/self.indices.count(action_index)
                        output_projection = tf.convert_to_tensor(output_projection, dtype=tf.float32)

                        x_ = tf.tensordot(x_, output_projection, axes = 1)
                        output = tf.add(output, x_)

            output = tf.nn.tanh(output)
        return output

class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, 128)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, 128)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
