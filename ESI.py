"""
Summary: 
    Evolutionary Selective Imitation. 
    Uses BipedalWalker-V3 as environment.
    Code for the paper found at - https://arxiv.org/abs/2009.08403.
Author: R. E.
Date: 20 September 2020
"""

import torch, numpy, os, datetime, gym, random, math, pandas, argparse, sys
from scipy.stats.stats import pearsonr
import matplotlib.pylab as plt
import numpy as np
from itertools import count
from collections import namedtuple
import multiprocessing as mp
from time import time
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, random_split, DataLoader
from torch import Tensor
from torch.nn import Linear, ReLU, Tanh, Sigmoid, Module, BCELoss, MSELoss, L1Loss
from torch.optim import SGD
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import pickle

# constants
MAX_INT = np.int32(2**31-1)
ENV_INPUTS = 24
STARTING_TRAJ_LEN = 5
TRAJLEN_INCREMENT = 5
MAX_TRAJREPO_LEN = 1000
MAX_NUM_EPISODES = int(10**7)
TOP_TRAJECTORY_NUM = 250
TRAIN_REWARD_THREASHOLD = 298
RUNS_TO_TEST_SOLVED = 100
REWARD_FOR_SOLVED = 300
REVERSED = -1
STEPS_RANDOM_TRAJ = 3000 

# global variable that counts overall number of steps
g_timestep_counter = 0

# settings
np.set_printoptions(precision=3)
gym.logger.set_level(40) 

def main():
    """
    Summary: Main
    """

    # init immitator, the class that trains the NN to imitate the active set
    nnImitator = NN_Imitator()

    # initiate the algorithm
    algo = ESI(nnImitator=nnImitator, gymEnvName='BipedalWalker-v3')

    # prints this execution's configuration.
    print (datetime.date.today(), "\n", vars(nnImitator), "\n", vars(algo), "\n", flush=True)

    # init the NN that imitates the data
    m = MLP(ENV_INPUTS)

    # generate a random trajectory with the random model
    reward, traj_1d, trajRewards_1d = algo.evaluateAgent(m, STEPS_RANDOM_TRAJ, 
            numpy.random.randint(100000), render=False)
    randomTrajObj = Traj(traj_1d, reward, m, [], trajRewards_1d)

    # run ESI with the random trajectory obejct
    algo.improveTrajectory(randomTrajObj)

class Traj():
    """
    Summary: an object that holds all the relevant information about an agent trajectory 

    @param traj_1d (numpy array): the actual trajectory - observation, action for every step
    @param reward (int): the reward of the trajectory
    @param agent (pytorch object): the model that made the trajectory
    @param trainingSample (numpy array): the samples the model trained to imitate
    @param trajRewards_1d (numpy array): reward for every step
    """
    
    def __init__(self, traj_1d, reward, agent, trainingSample, trajRewards_1d):
        
        # the actual trajectory - observation, action and reward for every step
        self.traj_1d=numpy.array(traj_1d)
        self.trajRewards_1d=numpy.array(trajRewards_1d)

        # the reward of the trajectory
        self.reward=reward
        
        # the model that made the trajectory
        self.agent=agent
        
        # the samples the model trained to imitate
        self.trainingSample=trainingSample

class ESI():
    """
    Summary: ESI algorithm

    @param nnImitator (NN_Imitator object): init immitator, the class that trains the NN 
        to imitate the active set
    @param gymEnvName (string): the name of the environment ESI would run on
    @param samplesPerTraj (int): number of samples that would be taken for imitation from every trajectory
    @param maxGameLength (int): limit maximum game length, in case of an endless loop
    @param sampleSize (int): size of set that is imitated by the random NN model
    """

    def __init__(self, nnImitator, gymEnvName, samplesPerTraj=125, maxGameLength=3000, sampleSize=25):

        # all trajectories that had ever been generated so far
        self.trajsToTryQueue = {}
        
        # save all trajectories that have been used to generate agents, for documentation purposes
        self.usedTrajs = []

        # count number of all samples executed so far
        self.sampleCounter = 0

        # size of set that is imitated by the random NN model
        self.sampleSize = sampleSize

        # init immitator, the class that trains the NN to imitate the active set
        self.nnImitator = nnImitator

        # the name of the environment ESI would run on
        self.gymEnvName = gymEnvName

        # number of samples that would be taken for imitation from every trajectory
        self.samplesPerTraj = samplesPerTraj

        # limit maximum game length, in case of an endless loop
        self.maxGameLength = maxGameLength

        # the scope of the initial few samples from which the samples are taken, 
        #    gradually increases as the algorithm progresses
        self.trajlen = STARTING_TRAJ_LEN
        

    def improveTrajectory(self, traj):
        """
        Summary: returns a sample (of length sampleSize) and model that achieved better score 
            than curReward

        @param traj (Traj object): the initial trajectory with which ESI starts to improve. Solves the 
            example given (bipedalwalker) here with a random initialization.
        """
        
        # set initial trajectory
        self.trajsToTryQueue[traj.reward] = traj

        pickleEvaledTrajRewardIds = []
        for i_episode in range(MAX_NUM_EPISODES):

            # get all rewards in the current trajectory pool sorted by value.
            sortedRewards_1d = numpy.sort(numpy.array(list(self.trajsToTryQueue.keys())))[::REVERSED]            
            print ("\n\nEpisode %d (%d samples). Cur queue: %s. Steps: %d" % (i_episode, self.sampleCounter, 
                sortedRewards_1d[:30], g_timestep_counter ), flush=True)
            
            # if it is found, pickle the agents with reward that is >300 on avg in 100 consecutive runs. 
            self._pickleSolution(sortedRewards_1d)

            # with open('%d_%d.pickle' % (startTime%1000000, sortedRewards_1d[0], ) ,'wb') as f: pickle.dump(self.trajsToTryQueue[sortedRewards_1d[0]], f)

            # to save RAM, delete all by the top few trajectories. The bottom ones aren't important.
            if len(sortedRewards_1d) > MAX_TRAJREPO_LEN:
                for r in sortedRewards_1d[MAX_TRAJREPO_LEN:]:    
                    del self.trajsToTryQueue[r]
            
            # add the top trajectories to the "pool" of potential trajectories from which the next 
            #    "bestTraj" will be picked
            trajPool_1d = []
            for r in sortedRewards_1d[:TOP_TRAJECTORY_NUM]:
                trajPool_1d.append(self.trajsToTryQueue[r])
            # pick next trajectory that will be used as best traj. As in, the trajectory from which the 
            #    model will imitate subsets.
            trajObj = self._pickNextTraj(trajPool_1d)

            # run an episode based on the trajectory object chosen. Add all the trajs made to the list
            #     of trajs to try.
            self._exhaustTraj(trajObj)
            # delete the traj we've 'exhausted' this episode
            del self.trajsToTryQueue[trajObj.reward]
            self.usedTrajs.append(trajObj)

            # increase the scope from which we sample 
            self.trajlen += TRAJLEN_INCREMENT

    def _pickleSolution(self, sortedRewards_1d):
        """
        Summary: of any good enough trajectory was made, try to run it 100 times to test if 
            it solved biped.

        @param sortedRewards_1d (numpy array): Rewards of all trajectories discovered so far 
            in any execution, sorted decreasingly
        """
        
        # iterate over all trajectories 
        for r in sortedRewards_1d[:TOP_TRAJECTORY_NUM]:
            # if trajectory train reward is below threshold, skip it.
            if (r < TRAIN_REWARD_THREASHOLD): continue
            # if already tested this trajectory (perhaps in previous episodes), don't run it 100 times again.
            if r in pickleEvaledTrajRewardIds: 
                continue
            pickleEvaledTrajRewardIds.append(r)

            # get trajectory
            curTraj = self.trajsToTryQueue[r]

            # get the trajectories' average reward over 100 consecutive rounds 
            totReward = 0
            for _ in range(RUNS_TO_TEST_SOLVED):
                reward, _, _ = self.evaluateAgent(curTraj.agent, self.maxGameLength, 
                    numpy.random.randint(100000), render=False)
                totReward += reward
            totReward = totReward / float(RUNS_TO_TEST_SOLVED)

            # if reward if >300 then biped is solved. Save the trajectory to disk with pickle.
            if (totReward >= REWARD_FOR_SOLVED):
                with open('%d_test%d_train%d.pickle' % (startTime%1000000, totReward, r) ,'wb') \
                    as f: pickle.dump(curTraj, f)        


    def _pickNextTraj(self, trajPool_1d):
        """
        Summary: pick the trajectory to imitate for the next episode by the one with the highest 
            reward sum in all the first samples up to the current scope length.

        @param trajPool_1d (numpy array): List of pool trajectories from which one will be chosen for 
            the next episode

        @return besttraj (numpy array): the trajectory chosen from the pool
        """
        
        # init 
        maxScopedReward = -MAX_INT
        besttraj = trajPool_1d[0]

        # iterate over pool and set best traj to the one with the highest (Scoped) reward sum
        for trajobj in trajPool_1d:
            if numpy.sum(trajobj.trajRewards_1d[:self.trajlen]) > maxScopedReward:
                maxScopedReward = numpy.sum(trajobj.trajRewards_1d[:self.trajlen])
                besttraj = trajobj

        return besttraj 

    def _exhaustTraj(self, trajObj):
        """
        Summary: Sample from the given trajectory many times, each time imitating the data sampled with 
            a model, executing the trained model and saving the results

        @param trajObj (numpy array): trajectory to sample from
        """
        
        for i_sample in range(self.samplesPerTraj):
            
            # sample from the trajectory 
            sampleIds_1d = numpy.random.randint(low=0, high=min(len(trajObj.traj_1d), self.trajlen), 
                size=int(self.sampleSize))
            
            # immitate the sample
            curModel, imitationCorr = self.nnImitator.gen_imitated_agent(MLP(ENV_INPUTS), 
                trajObj.traj_1d[sampleIds_1d])

            # run the model that is the result of the imitation
            reward, traj_1d, trajRewards_1d = self.evaluateAgent(curModel, self.maxGameLength, 
                numpy.random.randint(100000), render=False)
            
            # save results
            self.trajsToTryQueue[reward] = Traj(traj_1d, reward, curModel, 
                numpy.array(trajObj.traj_1d)[sampleIds_1d], trajRewards_1d)

            # increment sample counter
            self.sampleCounter += 1


    def evaluateAgent(self, model, steps, seed, render=False):
        """
        Summary: Run a model on an environment for a single trajectory 

        @param model (pytorch object): the NN agent to evaluate
        @param steps (int): number of steps to make in the environment
        @param seed (int): random seed
        @param render (boolean): whether to show the graphics of the robot or not
        
        @return total_reward (int): reward sum of the agent on the trajectory
        @return trajectory_1d (numpy array): trajectory (obervation, action pairs) of agent
        @return trajRewards_1d (numpy array): reward for each step in the trajectory 
            (same and matching indices as trajectory_1d)
        """

        global g_timestep_counter

        # create environment
        env = gym.make(self.gymEnvName)

        # init
        obs_1d = env.reset();
        total_reward = 0
        trajectory_1d = []
        trajRewards_1d = []

        # environment loop
        for stepIndex in range(steps):

            # render to screen according to parameter
            if (render): env.render()

            # conver to tensor to get action from model, and then back to numpy to make gym env step.
            action = model(Tensor(obs_1d)).detach().numpy()
            # add the observation and action to the trajectory
            trajectory_1d.append((obs_1d, action))
            # take the environment step
            obs_1d, reward, done, info = env.step(action)

            # add the reward and sum it
            trajRewards_1d.append(reward)
            total_reward += reward

            # exit if game is over
            if (done): break

        # increment total timestep counter
        g_timestep_counter += len(trajRewards_1d)
        
        return total_reward, trajectory_1d, trajRewards_1d

#######################################################################################
############################# NN ######################################################
#######################################################################################

class NN_Imitator():
    """
    Summary: Trains the NN agent to imitate the active set

    @param backpropIters (int): number of back propogration iterations to train the agent
    @param learningRate (float): learning rate of gradient descent
    @param batchSize (int): number of samples in every batch in training
    @param momentum (float): momentum of the gradient descent
    """

    def __init__(self, backpropIters=200, learningRate=(0.01/1.6), batchSize=15, momentum=0.9):

        # number of back propogration iterations to train the agent
        self.backpropIters=backpropIters
        # learning rate of gradient descent
        self.learningRate=learningRate
        # number of samples in every batch in training
        self.batchSize=batchSize
        # momentum of the gradient descent
        self.momentum=momentum

    def gen_imitated_agent(self, model, trajectory):
        """
        Summary: given a model and a trajectory, imitate the model to the trajectory

        @param model (pytorch object): random model to imitate the trajectory
        @param trajectory (numpy array): trajectory (obervation, action pairs) of agent

        @return model (pytorch object): the trained model after it imtiated the data
        @return imitationCorr (float): how good the imitation was as describe in the correlation 
            between the training data and the trained model estimation of the training data
        """
        
        # convert trajectory to two numpy arrays for the observation and actions
        trajectory = list(trajectory)
        obs_2d, actions_2d = zip(*trajectory) if (trajectory != []) else ([], [])
        obs_2d, actions_2d = numpy.array(obs_2d), numpy.array(actions_2d)
    
        # first run trajectory is empty so no need to train model
        if (len(obs_2d) == 0): 
            raise Exception('Empty trajectory')

        # prepare the data and train model
        train_dl = CSVDataset(obs_2d, actions_2d)
        train_dl = DataLoader(train_dl, batch_size=self.batchSize, shuffle=True)
        imitationCorr = self._train_model(model, train_dl)

        return model, imitationCorr

    def _train_model(self, model, train_dl):
        """
        Summary: use imitation learning to train the model on the training data

        @param model (pytorch object): random model to imitate the trajectory
        @param train_dl (pytorch dataloader): training data for the model to imitate

        @return corrToData (float): how good the imitation was as describe in the correlation 
            between the training data and the trained model estimation of the training data
        """
        
        # init optimization
        criterion = L1Loss()
        optimizer = SGD(model.parameters(), lr=self.learningRate, momentum=self.momentum)

        # start optimization cycle
        for _ in range(self.backpropIters):
            for i, (inputs, targets) in enumerate(train_dl):
                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                yhat = model(inputs)
                # calculate loss
                loss = criterion(yhat, targets)
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()

        # evalute the agent on its own training set to measure effectiveness of imitation by 
        #    pearson correlation
        corrToData = calcActionsSimilarity(
            model(torch.from_numpy(train_dl.dataset.X)).cpu().detach().numpy(), train_dl.dataset.y)    

        return corrToData

def calcActionsSimilarity(y1_2d, y2_2d):
    """
    Summary: pearson correlation between two 2d vectors by mean of 1d correlations

    @param y1_2d (numpy array): 2d vector
    @param y2_2d (numpy array): 2d vector

    @return corrMean (float): mean of pearson correlation between all 1d vectors.
    """

    # calculate mean of correlation of each 1d vectors (assuming 2d are matching by index)
    corrSum = 0
    for i in range(y1_2d.shape[1]):
        corrSum += corr(y1_2d[:, i], y2_2d[:, i])
    corrMean = corrSum / float(y1_2d.shape[1])

    return corrMean

def corr(v1, v2):
    """
    Summary: pearson correlation

    @param y1_2d (numpy array): 1d vector
    @param y2_2d (numpy array): 1d vector

    @return correlation (float): pearsonr
    """

    if len(v1) != len(v2): raise Exception("Arrays do not share same length.")

    # convert vectors to number arrays
    v1, v2 = numpy.array(v1).astype(float), numpy.array(v2).astype(float)

    # it's invalid to claculate correlations between two all-zero vectors
    if (v1 == v1[0]).all() or (v2 == v2[0]).all():
        return 0

    # pearsonr faster than numpy's corrcoef
    correlation = pearsonr(v1, v2)[0] #numpy.corrcoef(v1, v2)[0][1]

    return correlation
 
class CSVDataset(Dataset):
    """
    Summary: loads the dataset for the pytorch optimization

    @param x_2d (numpy array): observations
    @param y_2d (numpy array): labels
    """
    
    def __init__(self, x_2d, y_2d):

        self.X = x_2d.astype('float32')
        self.y = y_2d.astype('float32')

    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

class MLP(Module):
    """
    Summary: 1 hidden layer NN

    @param n_inputs (int): number of inputs in the current environment
    """
   
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 40)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='tanh')
        self.act1 = Tanh()

        # second hidden layer
        self.hidden2 = Linear(40, 40)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='tanh')
        self.act2 = Tanh()

        # third hidden layer and output
        self.hidden3 = Linear(40, 4)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Tanh()

 
    
    def forward(self, X):
        """
        Summary: forward propagate input

        @param X (pytorch object): observation input batch (2d)

        @return X (pytorch object): input after all the neuralnet transofrmations, 
            i.e the NN estimation.
        """

        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)

        return X


if __name__ == "__main__":
    startTime = time()
    try:
        main()
    finally:
        print ("\nTook: %.3f s" % (time() - startTime))


