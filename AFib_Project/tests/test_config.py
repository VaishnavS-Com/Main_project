"""Quick test to verify config.py loads correctly with Python version guard."""
import sys
sys.path.insert(0, '.')

from config.config import PYTHON_REQUIRED, BASE_DIR, SAMPLING_FREQUENCY, TOTAL_SUBJECTS

print('=' * 50)
print('  CONFIG LOADED SUCCESSFULLY')
print('=' * 50)
print(f'  Python Required : {".".join(map(str, PYTHON_REQUIRED))}')
print(f'  Python Running  : {sys.version.split()[0]}')
print(f'  Base Directory  : {BASE_DIR}')
print(f'  Sampling Freq   : {SAMPLING_FREQUENCY} Hz')
print(f'  Total Subjects  : {TOTAL_SUBJECTS}')
print('  Version guard   : PASSED')
print('=' * 50)
