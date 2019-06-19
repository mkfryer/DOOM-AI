from vizdoom import *
import random
import time
import numpy as np                         
from skimage import transform
from collections import deque
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf     
from Memory import Memory
from DQNet import DQN

warnings.filterwarnings('ignore')

""" """
def set_up_game():
    #game set up
    game = DoomGame()
    game.load_config("game_API/scenarios/defend_the_line.cfg")
    game.set_doom_scenario_path("game_API/scenarios/defend_the_line.wad")
    game.init()
    #actions
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    possible_actions = [shoot, left, right]
    return game, possible_actions

def preprocess_frame(frame):
    #turn to grey scale
    frame = np.mean(frame, axis=0)
    # Crop the screen (remove part that contains no information)
    cropped_frame = frame[15:-5,20:-20]
    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0
    # Resize
    preprocessed_frame = transform.resize(cropped_frame, [100,120])
    return preprocessed_frame # 100x120x1 frame

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((100,120), dtype=np.int) for i in range(stack_size)], maxlen=4)
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    return stacked_state, stacked_frames


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions, sess):
    exp_exp_tradeoff = np.random.rand()
    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(possible_actions)
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
                
    return action, explore_probability


def train_agent(training, memory, game, DQNetwork, batch_size, stacked_frames, saver):
    """ """


    if training == True:
        with tf.Session() as sess:
            # Initialize the variables
            sess.run(tf.global_variables_initializer())
            # Initialize the decay rate (that will use to reduce epsilon) 
            decay_step = 0
            # Init the game
            game.init()
            for episode in range(total_episodes):
                # Set step to 0
                step = 0
                # Initialize the rewards of the episode
                episode_rewards = []
                # Make a new episode and observe the first state
                game.new_episode()
                state = game.get_state().screen_buffer
                # Remember that stack frame function also call our preprocess function.
                state, stacked_frames = stack_frames(stacked_frames, state, True)
                while step < max_steps:
                    print("step", step)
                    step += 1
                    # Increase decay_step
                    decay_step +=1
                    # Predict the action to take and take it
                    action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions, sess)
                    # Do the action
                    reward = game.make_action(action)
                    # Look if the episode is finished
                    done = game.is_episode_finished()
                    # Add the reward to total reward
                    episode_rewards.append(reward)
                    # If the game is finished
                    if done:
                        # the episode ends so no next state
                        next_state = np.zeros((3, 84,84), dtype=np.int)
                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                        # Set step = max_steps to end the episode
                        step = max_steps
                        # Get the total reward of the episode
                        total_reward = np.sum(episode_rewards)
                        print('Episode: {}'.format(episode),
                                'Total reward: {}'.format(total_reward),
                                'Training loss: {:.4f}'.format(loss),
                                'Explore P: {:.4f}'.format(explore_probability))
                        memory.add((state, action, reward, next_state, done))
                    else:
                        # Get the next state
                        next_state = game.get_state().screen_buffer
                        # Stack the frame of the next_state
                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                        # Add experience to memory
                        memory.add((state, action, reward, next_state, done))
                        # st+1 is now our current state
                        state = next_state

                    ### LEARNING PART            
                    # Obtain random mini-batch from memory
                    batch = memory.sample(batch_size)
                    states_mb = np.array([each[0] for each in batch], ndmin=3)
                    actions_mb = np.array([each[1] for each in batch])
                    rewards_mb = np.array([each[2] for each in batch]) 
                    next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                    dones_mb = np.array([each[4] for each in batch])

                    target_Qs_batch = []
                    # Get Q values for next_state 
                    Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})
                    # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                    for i in range(0, len(batch)):
                        terminal = dones_mb[i]
                        # If we are in a terminal state, only equals reward
                        if terminal:
                            target_Qs_batch.append(rewards_mb[i])
                        else:
                            target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                            target_Qs_batch.append(target)
                            
                    targets_mb = np.array([each for each in target_Qs_batch])
                    loss, _ = sess.run(
                        [DQNetwork.loss, 
                        DQNetwork.optimizer], 
                        feed_dict={DQNetwork.inputs_: states_mb,
                        DQNetwork.target_Q: targets_mb,
                        DQNetwork.actions_: actions_mb})

                    # Write TF Summaries
                    summary = sess.run(
                        write_op, 
                        feed_dict={DQNetwork.inputs_: states_mb,
                        DQNetwork.target_Q: targets_mb,
                        DQNetwork.actions_: actions_mb})
                
                    writer.add_summary(summary, episode)
                    writer.flush()
                # Save model every 5 episodes
                if episode % 5 == 0:
                    save_path = saver.save(sess, "./models/model.ckpt")
                    print("Model Saved")


def watch_play(saver, stacked_frames):
    with tf.Session() as sess:
        game, possible_actions = set_up_game()
        totalScore = 0
        # Load the model
        saver.restore(sess, "./models/model.ckpt")
        game.init()
        for i in range(1):
            done = False
            game.new_episode()
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)
                
            while not game.is_episode_finished():
                # Take the biggest Q value (= the best action)
                Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
                # Take the biggest Q value (= the best action)
                choice = np.argmax(Qs)
                action = possible_actions[int(choice)]
                game.make_action(action)
                done = game.is_episode_finished()
                score = game.get_total_reward()
                if done:
                    break  
                else:
                    print("else")
                    next_state = game.get_state().screen_buffer
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    state = next_state
                    
            score = game.get_total_reward()
            print("Score: ", score)
        game.close()


if __name__ == "__main__":

    game, possible_actions = set_up_game()
    stack_size = 4
    # Initialize deque with zero-images one array for each image
    stacked_frames  =  deque([np.zeros((100,120), dtype=np.int) for i in range(stack_size)], maxlen=4) 
    ### MODEL HYPERPARAMETERS
    state_size = [100,120,4]      # Our input is a stack of 4 frames hence 100x120x4 (Width, height, channels) 
    learning_rate =  0.00025      # Alpha (aka learning rate)
    ### TRAINING HYPERPARAMETERS
    total_episodes = 2         # Total episodes for training
    max_steps = 20              # Max possible steps in an episode
    batch_size = 64             
    # FIXED Q TARGETS HYPERPARAMETERS 
    max_tau = 10000 #Tau is the C step where we update our target network
    # EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start
    explore_stop = 0.01            # minimum exploration probability 
    decay_rate = 0.00005            # exponential decay rate for exploration prob
    # Q LEARNING hyperparameters
    gamma = 0.95               # Discounting rate
    ### MEMORY HYPERPARAMETERS
    ## If you have GPU change to 1million
    pretrain_length = 500   # Number of experiences stored in the Memory when initialized for the first time
    memory_size = 500     # Number of experiences the Memory can keep
    ### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
    training = True
    ## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
    episode_render = False
    # Reset the graph
    tf.reset_default_graph()
    # Instantiate the DQNetwork
    DQNetwork = DQN(state_size, len(possible_actions), learning_rate, "")

    # Instantiate memory
    memory = Memory(max_size = memory_size)
    # Render the environment
    game.new_episode()
    for i in range(pretrain_length):
        # print("step:", i)
        # If it's the first step
        if i == 0:
            # First we need a state
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        # Random action
        action = random.choice(possible_actions)
        # Get the rewards
        reward = game.make_action(action)
        done = game.is_episode_finished()
        if  done:
            # We finished the episode
            next_state = np.zeros(state.shape)
            # Add experience to memory
            memory.add((state, action, reward, next_state, done))
            # Start a new episode
            game.new_episode()
            # First we need a state
            state = game.get_state().screen_buffer
            # Stack the frames
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        else:
            # Get the next state
            next_state = game.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            # Add experience to memory
            memory.add((state, action, reward, next_state, done))
            # Our state is now the next_state
            state = next_state


    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter("/tensorboard/dqn/1")
    ## Losses
    tf.summary.scalar("Loss", DQNetwork.loss)
    write_op = tf.summary.merge_all()
    # Saver will help us to save our model
    saver = tf.train.Saver()
    train_agent(training, memory, game, DQNetwork, batch_size, stacked_frames, saver)
    watch_play(saver, stacked_frames)











# def play():
#     game = DoomGame()
#     game.load_config("game_API/scenarios/defend_the_line.cfg")
#     game.init()

#     shoot = [0, 0, 1]
#     left = [1, 0, 0]
#     right = [0, 1, 0]
#     actions = [shoot, left, right]

#     episodes = 10
#     for i in range(episodes):
#         game.new_episode()
#         while not game.is_episode_finished():
#             state = game.get_state()
#             img = state.screen_buffer
#             misc = state.game_variables
#             reward = game.make_action(random.choice(actions))
#             print ("\treward:", reward)
#             time.sleep(0.02)
#         print ("Result:", game.get_total_reward())
#         time.sleep(2)