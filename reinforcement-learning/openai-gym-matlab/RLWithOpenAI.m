%% OpenAI Cart-Pole Solver 
% By Paulo Carvalho
% Based on: https://www.mathworks.com/help/reinforcement-learning/ug/train-dqn-agent-to-balance-cart-pole-system.html
clear;clc;

%% Script settings
doTraining = true; % If set to true, the script performs training

%% Create the environment for training
env = OpenAIWrapper();

rng(0);

%% Define the neural networks 
statePath = [
    imageInputLayer([4 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(24,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(24,'Name','CriticStateFC2')];
actionPath = [
    imageInputLayer([1 1 1],'Normalization','none','Name','action')
    fullyConnectedLayer(24,'Name','CriticActionFC1')];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','output')];
criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);    
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

criticOpts = rlRepresentationOptions('LearnRate',0.001,'GradientThreshold',1);

%% Define the learning agent
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'state'},'Action',{'action'},criticOpts);

agentOpts = rlDQNAgentOptions(...
    'UseDoubleDQN',false, ...    
    'TargetUpdateMethod',"periodic", ...
    'TargetUpdateFrequency',4, ...   
    'ExperienceBufferLength',100000, ...
    'DiscountFactor',0.995, ...
    'MiniBatchSize',128);

agent = rlDQNAgent(critic,agentOpts);

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 1000, ...
    'MaxStepsPerEpisode', 200, ...
    'Verbose', false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'ScoreAveragingWindowLength',100, ...
    'StopTrainingValue',195); 


%% Perform training (if required) or load a saved agent
if doTraining    
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load pretrained agent for the example.
    load('MATLABCartpoleDQN.mat','agent');
end

%% Simulate the environment with the trained agent
simOptions = rlSimulationOptions('MaxSteps',200);
experience = sim(env,agent,simOptions);

totalReward = sum(experience.Reward)