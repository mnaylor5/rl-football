# Python scripts for scheduling training scenarios
import gfootball.env as football_env
from collections import deque 
import random 

SCENARIOS = {
    'basic':[
        'academy_empty_goal_close',
        'academy_empty_goal',
        'academy_run_to_score',
        'academy_run_to_score_with_keeper',
        'academy_pass_and_shoot_with_keeper',
    ],
    'easy':[
        'academy_counterattack_easy',
        'academy_run_pass_and_shoot_with_keeper',
        'academy_3_vs_1_with_keeper',
        'academy_single_goal_versus_lazy',
        'academy_corner'
    ],
    'medium':[
        '11_vs_11_easy_stochastic',
        'academy_counterattack_hard'
    ],
    'hard':[
        '11_vs_11_stochastic',
        '11_vs_11_hard_stochastic'
    ],
    'full':['11_vs_11_kaggle']
}

class TrainingPlan:
    def __init__(self, 
                 basic_rounds, 
                 easy_rounds,
                 medium_rounds,
                 hard_rounds,
                 full_match_rounds,
                 representation='extracted',
                 scenario_map=SCENARIOS):
        
        self.rounds = {
            'basic':basic_rounds,
            'easy':easy_rounds,
            'medium':medium_rounds,
            'hard':hard_rounds,
            'full':full_match_rounds
        }
        self.scenario_map = scenario_map
        self.representation=representation
        self.schedule_training()
        self.current_scenario_name = self.training_plan[0]
    
    def schedule_training(self):
        self.training_plan = deque()
        for difficulty in self.rounds.keys():
            for _ in range(self.rounds[difficulty]):
                scenario = random.choice(self.scenario_map[difficulty])
                self.training_plan.append(scenario)
    
    def get_next(self):
        scenario = self.training_plan.popleft()
        self.current_scenario_name = scenario
        env = football_env.create_environment(env_name=scenario, 
                                              stacked=False, 
                                              logdir='/tmp/football', 
                                              representation=self.representation,
                                              write_goal_dumps=False, 
                                              write_full_episode_dumps=False, 
                                              render=False)
        return env