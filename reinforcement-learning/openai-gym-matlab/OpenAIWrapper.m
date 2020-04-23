classdef OpenAIWrapper < rl.env.MATLABEnvironment
    %OPENAIWRAPPER: Wrapper for Python OpenAI Gym for RL trainig in MATLAB
    
    properties
        open_env = py.gym.make('CartPole-v1'); % Select environment name here  
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        function this = OpenAIWrapper()
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([4 1]);
            ObservationInfo.Name = 'CartPoleObservation';
            ObservationInfo.Description = 'Position,Velocity,Angle,VelocityAtTip';
            
            % Continous Action Specification
            ActionInfo = rlFiniteSetSpec([0 1]); 
            ActionInfo.Name = 'PushingPole';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
        end
        
        % Apply system dynamics and simulate the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)
            % Call OpenAI Step Function
            result = this.open_env.step(int16(Action)); % Type casting here!
            
            % Collect Results
            Observation = double(result{1})'; % Type casting here!
            Reward = result{2};
            IsDone = result{3};
            LoggedSignals = [];
                 
        end
        
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
            result = this.open_env.reset();
            InitialObservation = double(result)'; % Type casting here!
        end
    end
end
