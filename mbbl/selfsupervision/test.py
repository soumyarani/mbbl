import numpy as np
import tensorflow as tf
from encoder import Encoder
from dynamics import Dynamics

state_dim = 50
action_dim = 12
feature_dim = 20
encoder_hidden_sizes = [128, 64]
dynamics_hidden_sizes = [64, 32]
encoder_lr = 0.01
dynamics_lr = 0.01

tf.reset_default_graph()

with tf.Session() as sess:

    encoder = Encoder(sess=sess,
                    input_dim=state_dim,
                    output_dim=feature_dim,
                    gamma=0.98,
                    hidden_sizes=encoder_hidden_sizes,
                    learning_rate=encoder_lr)

    dynamics = Dynamics(sess=sess,
                    state_dim=feature_dim,
                    action_dim=action_dim,
                    hidden_sizes=dynamics_hidden_sizes,
                    learning_rate=dynamics_lr)

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./output", sess.graph)

    ############################ Test Encoder ######################################
    state_1 = np.random.rand(state_dim)
    state_2 = np.random.rand(state_dim)
    reward_1 = np.random.rand(1)
    reward_2 = np.random.rand(1)
    wasserstein = np.random.rand(1)

    encoded_1 = encoder.predict(state=state_1)

    print("Before:" , encoded_1)

    encoder.update(state_1, state_2, reward_1, reward_2, wasserstein)

    encoded_1 = encoder.predict(state=state_1)

    print("After:" , encoded_1)
    #################################################################################

    ############################ Test Dynamics ######################################
    state = np.random.rand(state_dim)
    next_state = np.random.rand(state_dim)
    action = np.random.rand(action_dim)

    encoded_state = encoder.predict(state=state)
    encoded_next_state = encoder.predict(state=next_state)

    next_state_from_dynamics = dynamics.predict(state=encoded_state, action=action)

    print("Before:" , next_state_from_dynamics)

    dynamics.update(state=encoded_state, action=action, next_state=encoded_next_state)

    next_state_from_dynamics = dynamics.predict(state=encoded_state, action=action)

    print("After:" , next_state_from_dynamics)
    #################################################################################