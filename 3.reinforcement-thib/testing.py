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


#number ghosts, change number training -x, 

#python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic
command = f"python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic -q "
# Run the command and collect output
result = subprocess.run(command, stdout=subprocess.PIPE, shell=True)


print(type(result.stdout))


# Sample output as a byte string
byte_output = (b'Beginning 50 episodes of Training\nTraining Done (turning off epsilon and alpha)\n'
               b'---------------------------------------------\nPacman emerges victorious! Score: 1307\n'
               b'Pacman emerges victorious! Score: 1325\nPacman emerges victorious! Score: 1313\n'
               b'Pacman emerges victorious! Score: 1308\nPacman emerges victorious! Score: 1324\n'
               b'Pacman emerges victorious! Score: 1328\nPacman emerges victorious! Score: 1313\n'
               b'Pacman died! Score: -24\nPacman emerges victorious! Score: 1334\n'
               b'Pacman emerges victorious! Score: 1313\nAverage Score: 1184.1\nScores:        1307.0, 1325.0, '
               b'1313.0, 1308.0, 1324.0, 1328.0, 1313.0, -24.0, 1334.0, 1313.0\nWin Rate:      9/10 (0.90)\n'
               b'Record:        Win, Win, Win, Win, Win, Win, Win, Loss, Win, Win\n')

# Use regular expression to find all instances of 'Score: <number>'
# The pattern is looking for the word 'Score:', followed by optional spaces, then a dash or digits ending with a digit
# The 'b' before the string literal indicates a byte string pattern
pattern = b'Score:\s*(-?\d+)'

# Find all matches in the byte output
scores = re.findall(pattern, result.stdout)

# Print each score
for score in scores:
    print('Score:', score.decode()) 







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
