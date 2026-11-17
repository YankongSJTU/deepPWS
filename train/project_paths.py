from pathlib import Path
import os
PROJECT_ROOT = Path(os.environ.get('FACEBA_PROJECT_ROOT', str(Path(__file__).resolve().parents[2]))).expanduser().resolve()
