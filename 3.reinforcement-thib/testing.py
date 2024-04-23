import subprocess
import re
from scipy.stats import ttest_rel
import random
 

def run_pacman(agent_type, num_training, num_games, layout, ghost_type, num_ghosts):
    # Build the command to run the pacman game with specified arguments
    command = f"python pacman.py -p {agent_type} -a numTraining={num_training} -x {num_training} -n {num_games} -l {layout} -g {ghost_type} -k {num_ghosts} -q"
    # Run the command and collect output
    result = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    # Extract scores from the output using regular expressions
    scores = re.findall(r"Average Score: (\-?\d+\.?\d*)", str(result.stdout))
    # Convert the scores to floats
    scores = [float(score) for score in scores]
    return scores


#python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic
command = f"python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic -q "
# Run the command and collect output
result = subprocess.run(command, stdout=subprocess.PIPE, shell=True)


print(type(result))

"""
def run_tests():
    # Define the different agents, environments, and other parameters
    agents = ['SemiGradTD', 'TrueOnlineTD', 'ApproximateQAgent']
    environments = ['mediumClassic', 'smallGrid']  # Add more environments as needed
    num_runs = 100  # Number of runs for averaging scores
    test_scores = {agent: [] for agent in agents}

    # Run games for each agent and environment
    for environment in environments:
        for agent in agents:
            scores = []
            for _ in range(num_runs):
                num_ghosts = random.choice(range(1, 5))  # Randomize number of ghosts
                scores += run_pacman(agent, 50, 60, environment, 'RandomGhost', num_ghosts)
            test_scores[agent].append(scores)

    return test_scores


def perform_statistical_analysis(test_scores):
    for agent1 in test_scores:
        for agent2 in test_scores:
            if agent1 != agent2:
                for env_index, environment in enumerate(environments):
                    # Perform a paired t-test for each environment
                    t_stat, p_value = ttest_rel(test_scores[agent1][env_index], test_scores[agent2][env_index])
                    print(f"Paired t-test between {agent1} and {agent2} in {environment}: t-statistic={t_stat}, p-value={p_value}")


def main():
    test_scores = run_tests()
    perform_statistical_analysis(test_scores)

if __name__ == "__main__":
    main()
"""