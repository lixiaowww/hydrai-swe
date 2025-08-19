from neuralhydrology import nh_run
from pathlib import Path
import sys

# Add the neuralhydrology src to the python path
sys.path.append('src/neuralhydrology')
from prepare_data import prepare_data_for_neuralhydrology


def train_model_with_neuralhydrology():
    """
    Trains a model using the NeuralHydrology library.
    """
    print("Training model with NeuralHydrology...")

    # --- 1. Prepare data ---
    # This step converts our processed data into the format required by NeuralHydrology.
    processed_data_dir = "data/processed"
    neuralhydrology_data_dir = "src/neuralhydrology/data"
    prepare_data_for_neuralhydrology(processed_data_dir, neuralhydrology_data_dir)

    # --- 2. Train model ---
    # The training is now driven by the config file.
    config_path = Path("src/neuralhydrology/config.yml")
    
    # Change to NeuralHydrology directory for training
    import os
    original_dir = os.getcwd()
    neuralhydrology_dir = Path("src/neuralhydrology")
    
    try:
        os.chdir(neuralhydrology_dir)
        print(f"Changed to directory: {os.getcwd()}")
        
        # Load configuration
        from neuralhydrology.utils.config import Config
        cfg = Config(Path(config_path.name))  # Use Path object
        
        # Start training
        nh_run.start_training(cfg)
        
    finally:
        # Return to original directory
        os.chdir(original_dir)
        print(f"Returned to directory: {os.getcwd()}")
    
    print("NeuralHydrology model training finished.")
    print("You can find the trained model and results in the 'runs' directory.")


if __name__ == "__main__":
    train_model_with_neuralhydrology()