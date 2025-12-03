import torch
import pathlib
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# Monkey-patch PosixPath to work on Windows
class PosixPathWindows(pathlib.WindowsPath):
    """A wrapper to make PosixPath loadable on Windows"""
    pass

def convert_paths(obj):
    """Recursively convert Path objects to strings"""
    if isinstance(obj, pathlib.Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_paths(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_paths(item) for item in obj)
    return obj

def strip_checkpoint_params(checkpoint):
    """Remove conflicting hyperparameters from checkpoint"""
    if 'hyper_parameters' not in checkpoint:
        print("\n⚠ No 'hyper_parameters' key found in checkpoint")
        return checkpoint
    
    print("\nOriginal hyperparameters:")
    for key, value in checkpoint['hyper_parameters'].items():
        print(f"  {key}: {value}")
    
    # Keep only essential architecture parameters
    keep_params = [
        'num_symbols', 'num_speakers', 'resblock', 'resblock_kernel_sizes',
        'resblock_dilation_sizes', 'upsample_rates', 'upsample_initial_channel',
        'upsample_kernel_sizes', 'filter_length', 'hop_length', 'win_length',
        'mel_channels', 'mel_fmin', 'mel_fmax', 'inter_channels', 'hidden_channels',
        'filter_channels', 'n_heads', 'n_layers', 'kernel_size', 'p_dropout',
        'n_layers_q', 'use_spectral_norm', 'gin_channels', 'use_sdp', 'segment_size'
    ]
    
    keys_to_remove = [key for key in checkpoint['hyper_parameters'].keys() 
                      if key not in keep_params]
    
    removed = []
    for key in keys_to_remove:
        if key in checkpoint['hyper_parameters']:
            del checkpoint['hyper_parameters'][key]
            removed.append(key)
    
    if removed:
        print(f"\n✓ Removed conflicting parameters: {', '.join(removed)}")
    else:
        print("\n⚠ No conflicting parameters found to remove")
    
    print("\nRemaining hyperparameters:")
    for key, value in checkpoint['hyper_parameters'].items():
        print(f"  {key}: {value}")
    
    return checkpoint

def process_checkpoint():
    """Main processing function with file/folder pickers"""
    # Hide the root window
    root = tk.Tk()
    root.withdraw()
    
    # Select input checkpoint file
    print("Please select the checkpoint file to process...")
    input_checkpoint = filedialog.askopenfilename(
        title="Select Checkpoint File",
        filetypes=[("Checkpoint files", "*.ckpt"), ("All files", "*.*")]
    )
    
    if not input_checkpoint:
        print("No file selected. Exiting.")
        return
    
    # Select output folder
    print("Please select the output folder...")
    output_folder = filedialog.askdirectory(
        title="Select Output Folder"
    )
    
    if not output_folder:
        print("No output folder selected. Exiting.")
        return
    
    # Generate output filename
    input_filename = os.path.basename(input_checkpoint)
    output_filename = f"processed-{input_filename}"
    output_checkpoint = os.path.join(output_folder, output_filename)
    temp_checkpoint = os.path.join(output_folder, f"temp-{input_filename}")
    
    print(f"\n{'='*60}")
    print(f"Input file: {input_checkpoint}")
    print(f"Output file: {output_checkpoint}")
    print(f"{'='*60}\n")
    
    # Replace PosixPath with Windows-compatible version
    original_posix = pathlib.PosixPath
    pathlib.PosixPath = PosixPathWindows
    
    try:
        # Step 1: Load and convert checkpoint
        print("Step 1: Loading checkpoint...")
        checkpoint = torch.load(input_checkpoint, weights_only=False, map_location='cpu')
        print("✓ Checkpoint loaded successfully")
        
        print("\nStep 2: Converting path objects to strings...")
        checkpoint = convert_paths(checkpoint)
        print("✓ Path conversion complete")
        
        # Save temporary converted checkpoint
        print(f"\nStep 3: Saving temporary converted checkpoint...")
        torch.save(checkpoint, temp_checkpoint)
        print("✓ Temporary file saved")
        
        # Step 4: Strip parameters
        print("\nStep 4: Stripping conflicting parameters...")
        checkpoint = strip_checkpoint_params(checkpoint)
        
        # Restore original PosixPath
        pathlib.PosixPath = original_posix
        
        # Step 5: Save final checkpoint
        print(f"\nStep 5: Saving final processed checkpoint...")
        torch.save(checkpoint, output_checkpoint)
        print("✓ Final checkpoint saved successfully!")
        
        # Clean up temporary file
        if os.path.exists(temp_checkpoint):
            os.remove(temp_checkpoint)
            print("✓ Temporary file cleaned up")
        
        print(f"\n{'='*60}")
        print("✓ PROCESSING COMPLETE!")
        print(f"Your processed checkpoint is ready at:")
        print(f"{output_checkpoint}")
        print(f"{'='*60}\n")
        
        # Show success message
        messagebox.showinfo(
            "Success", 
            f"Checkpoint processed successfully!\n\nOutput saved to:\n{output_checkpoint}"
        )
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Restore original PosixPath even on error
        pathlib.PosixPath = original_posix
        
        # Clean up temporary file if it exists
        if os.path.exists(temp_checkpoint):
            os.remove(temp_checkpoint)
        
        messagebox.showerror("Error", f"An error occurred:\n\n{str(e)}")
    
    finally:
        root.destroy()

if __name__ == "__main__":
    print("Checkpoint Converter & Stripper")
    print("="*60)
    process_checkpoint()   
