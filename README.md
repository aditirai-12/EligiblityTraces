#EligiblityTraces
run
python testing_all_at_once.py
in the command line to run the test for all 3 agents at once and get the graph for it

run
python testing_ghosts.py
in the command line to run the test for 3 different number of ghosts at once and get the graph for it.

run

python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic

to get a single result for ApproximateQAgent, the number beside -x is the number of training games, the number beside -n is the number of testing games. After -l we can enter the layout.

Put SemiGradientTDAgent or TrueOnlineTDAgent instead of ApproximateQAgent in the above command to get results for these two agents we coded.

run

python testing_win_rates_tepisodes.py

to get the progression of winrate for different number of training games in a graph
