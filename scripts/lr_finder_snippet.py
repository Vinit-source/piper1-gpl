# =============================================================================
# OPTIONAL: OPTIMAL LEARNING RATE FINDER
# =============================================================================
# Copy this code into a new cell before the "START TRAINING" cell in your notebook.
# This code finds the optimal learning rate and updates the `config` object.

try:
    print("Running Learning Rate Finder...")
    import sys
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.tuner import Tuner
    from piper.train.vits.lightning import VitsModel
    from piper.train.vits.dataset import VitsDataModule

    # Ensure piper src is in path
    if f"{config.piper_dir}/src" not in sys.path:
        sys.path.append(f"{config.piper_dir}/src")

    # DataModule
    # We use the same parameters as the training loop
    csv_path = metadata_csv_path
    config_path = f"{config.local_output_dir}/{config.model_name}.json"

    datamodule = VitsDataModule(
        csv_path=csv_path,
        cache_dir=config.local_cache_dir,
        espeak_voice=config.espeak_voice,
        config_path=config_path,
        voice_name=config.model_name,
        batch_size=config.batch_size,
        num_workers=4,
        sample_rate=config.sample_rate,
        num_speakers=config.num_speakers,
        audio_dir=config.local_wavs_dir
    )

    # Model
    model = VitsModel(
        batch_size=config.batch_size,
        sample_rate=config.sample_rate,
        num_speakers=config.num_speakers,
        learning_rate=config.learning_rate,
        finetune_mode=config.finetune_mode,
        encoder_lr_scale=config.encoder_lr_scale,
        flow_lr_scale=config.flow_lr_scale,
        decoder_lr_scale=config.decoder_lr_scale,
        lr_scheduler_type=config.lr_scheduler_type,
        lr_patience=config.lr_patience,
        lr_factor=config.lr_factor,
        lr_min=config.lr_min,
        grad_clip=config.grad_clip,
    )

    # Load pretrained weights if applicable
    if config.use_pretrained:
        # Use the URL used in training or a default one for fine-tuning
        ckpt_url = "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/ljspeech/medium/lj-med_1000.ckpt"
        print(f"Loading pretrained weights from {ckpt_url} for LR finding...")
        
        # We use torch.hub to load from URL seamlessly
        state_dict = torch.hub.load_state_dict_from_url(ckpt_url, map_location=config.device)
        
        # Handle 'state_dict' key if present (Lightning checkpoints usually have 'state_dict')
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        # Load into model
        model.load_state_dict(state_dict, strict=False)

    # Trainer
    trainer = pl.Trainer(
        accelerator=config.device,
        devices=1,
        max_epochs=100,
        default_root_dir=config.local_output_dir,
        precision=config.precision
    )

    # Tuner
    tuner = Tuner(trainer)
    # Run LR finder
    lr_finder = tuner.lr_find(model, datamodule=datamodule, min_lr=1e-6, max_lr=1e-2, num_training=100)
    
    # Get suggestion
    suggested_lr = lr_finder.suggestion()
    print(f"Suggested Learning Rate: {suggested_lr}")
    
    # Update config automatically
    if suggested_lr:
        config.learning_rate = suggested_lr
        print(f"Updated config.learning_rate to {config.learning_rate}")
        
    # Plot results
    fig = lr_finder.plot(suggest=True)
    fig.show()

except Exception as e:
    print(f"LR Finder failed: {e}")
    print("Proceeding with default training configuration...")
