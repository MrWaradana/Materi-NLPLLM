
import transformers
print(f"Transformers version: {transformers.__version__}")
try:
    from transformers import modeling_layers
    print("transformers.modeling_layers exists")
except ImportError:
    print("transformers.modeling_layers DOES NOT exist")

import peft
print(f"Peft version: {peft.__version__}")
