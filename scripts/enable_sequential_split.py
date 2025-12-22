
import json
import os

NOTEBOOK_PATH = 'notebooks/piper_finetuning_en_IN_refac_sgm.ipynb'

def update_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: {NOTEBOOK_PATH} not found.")
        return

    try:
        with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error loading notebook: {e}")
        return

    cells = nb.get('cells', [])
    updated = False

    # Find the "START TRAINING" cell or where train_cmd is defined
    for cell in cells:
        source = cell.get('source', [])
        if isinstance(source, str):
            # Convert to list if string
            source = [s + '\n' for s in source.splitlines()]
            # Clean up newlines if double
            source = [s.replace('\n\n', '\n') for s in source]
        
        # We need to act on list of strings
        content = "".join(source)
        
        if "python -m piper.train fit" in content and "split_method" not in content:
            # We want to add --data.split_method sequential
            # We can add it after --data.num_workers 4, for example.
            
            new_source = []
            for line in source:
                new_source.append(line)
                if "--data.num_workers" in line:
                    # Add our line
                    # Preserve indentation? Check previous line indentation
                    indent = line[:line.find("--")]
                    if not indent: indent = "    " # Default
                    new_source.append(f"{indent}--data.split_method sequential \\\\\n")
            
            cell['source'] = new_source
            updated = True
            print("Updated training command with split_method sequential.")
            break
        elif "split_method" in content:
            print("Notebook already contains split_method.")
            return

    if updated:
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully modified {NOTEBOOK_PATH}")
    else:
        print("Could not find the training command to update.")

if __name__ == "__main__":
    update_notebook()
