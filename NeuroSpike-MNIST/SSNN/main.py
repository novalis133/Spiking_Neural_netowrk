import torch
from snntorch import backprop, surrogate
from config.config import NetworkConfig, TrainingConfig
from models.neurons import LeakySNN, AlphaSNN, SynapticSNN, LapicqueSNN
from training.trainer import Trainer
from utils.data_loader import DataManager
from utils.visualization import ResultsVisualizer
from config.experiments import EXPERIMENTS
from utils.analysis import ExperimentAnalyzer
from utils.checkpoint_manager import CheckpointManager
from utils.advanced_viz import AdvancedVisualizer
from utils.export import ResultExporter

def run_experiment(experiment_config, network_config, training_config):
    
    for model_name in experiment_config.models:
        ModelClass = globals()[model_name]
        model = ModelClass(network_config, spike_grad).to(device)
        trainer = Trainer(model, training_config, device)
        
        for bp_method_name in experiment_config.bp_methods:
            bp_method = getattr(backprop, bp_method_name)
            print(f"\nTraining {model_name} with {bp_method_name}")
            trainer.train(train_loader, test_loader, bp_method)

def main():
    # Configurations
    network_config = NetworkConfig()
    training_config = TrainingConfig()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loading
    data_manager = DataManager(training_config)
    train_loader, test_loader = data_manager.get_fashion_mnist()
    
    # Initialize visualizer
    visualizer = ResultsVisualizer('results')
    
    # Define models and backpropagation methods
    models = [LeakySNN, AlphaSNN, SynapticSNN, LapicqueSNN]
    bp_methods = [backprop.BPTT, backprop.RTRL, backprop.TBPTT]
    spike_grad = surrogate.atan()
    
    # Initialize analysis tools
    analyzer = ExperimentAnalyzer('results')
    checkpoint_mgr = CheckpointManager('checkpoints')
    adv_viz = AdvancedVisualizer('results')
    
    # Initialize exporters
    exporter = ResultExporter('exports')
    
    # Run experiments
    for exp_name, exp_config in EXPERIMENTS.items():
        print(f"\nRunning experiment: {exp_name}")
        run_experiment(exp_config, network_config, training_config)
        
        # Analyze results
        results = analyzer.load_experiment_results(exp_name)
        performance_df = analyzer.analyze_performance(results)
        
        # Generate visualizations
        plots = {
            'convergence': analyzer.compare_convergence_rates(results),
            'interactive_comparison': adv_viz.create_interactive_comparison(exp_name),
            'heatmap': adv_viz.create_heatmap_comparison(exp_name)
        }
        
        # Export results
        exporter.export_experiment(exp_name, results, performance_df, plots)
        
        # Clean old checkpoints
        checkpoint_mgr.clean_old_checkpoints(exp_name)

if __name__ == "__main__":
    main()