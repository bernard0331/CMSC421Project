import neat
import visualize
from EnvironmentHelper import is_dead, getProgress
from FrameHelper import FrameProcessor
from pynput.keyboard import Key, Controller

class NEATController:
    """Holds an instance of population"""

    def __init__(self, config_file, frame_processor):
        """Initializes a population using information provided in the config file.

        Args:
            config_file: The path of the config file for NEAT parameters
            frameprocessor: The frameprocessor for the window training should be ran on
        """
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
        self.pop = neat.Population(self.config)
        self.winner = None
        self.stats = None
        self.frame_processor = frame_processor
        self.keyboard = Controller()

        # Add a stdout reporter to show progress in the terminal.
        self.pop.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        self.pop.add_reporter(self.stats)
        self.pop.add_reporter(neat.Checkpointer(5))


    def show_model(self):
        """Shows output of the most fit genome against training data."""
        
        print('\nOutput:')
        visualize.draw_net(self.config, self.winner, True)
        visualize.draw_net(self.config, self.winner, True, prune_unused=True)
        visualize.plot_stats(self.stats, ylog=False, view=True)
        visualize.plot_species(self.stats, view=True)


    def eval(self, network = None):
        """Runs the winning network on the environment
        
        Args:
            network: the network to evaluate, uses the class' best network on default
        """
        if network == None and self.winner != None:
            network = neat.nn.FeedForwardNetwork.create(self.winner, self.config)
        elif network == None:
            raise ValueError("Network not found, run evolve first or provide a network")
        
        print("Starting Run")
        self.keyboard.press(Key.space)
        self.keyboard.release(Key.space)

        progress = 0.0
        img = self.frame_processor.get_frame()
        dead = False
        while not dead:
            prev_img = img
            img = self.frame_processor.get_frame()
            raw_img = self.frame_processor.get_raw_frame(35,180,330,500)
            
            progress = max(getProgress(raw_img, 13, 283, 6), progress)
            
            if is_dead(prev_img, img):
                break
            
            action = network.activate(img.reshape(-1))[0]
            if action >= 0.5:
                #Could make this hold space until next iteration
                self.keyboard.press(Key.space)
                self.keyboard.release(Key.space)
            
        print("Agent died")

        return progress

    def evolve(self, generations=10):
        """Runs evolution on the population"""
        print("Starting evolution")
        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                print("Genome id: {}".format(genome_id))
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                runscore = self.eval(net)
                genome.fitness = runscore

        self.winner = self.pop.run(eval_genomes, generations)

    def save_model(self, path):
        """Saves the model to a file"""
        pass
    
    def load_model(self, path):
        """Loads the model from a file"""
        pass

