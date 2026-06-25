import numpy as np, sys
sys.path.insert(0, '.')
from modules.config_loader import ConfigLoader
from modules.engine_ai import AIEngine
from modules.engine_traditional import TraditionalEngine
from modules.data_manager import DataManager
from modules.obstacle_classes import CLASS_NAMES

cfg = ConfigLoader('./config.yaml')
ea  = AIEngine(cfg)
et  = TraditionalEngine(cfg)
dm  = DataManager(cfg)
dm.load_session('session_train_007')

print('=== session_train_007 ===')
for fid in [0, 50, 100, 150, 200]:
    dm.seek(fid)
    fr = dm.get_current_frame()
    rt = et.process(fr)
    ra = ea.process(fr)
    print(f'frame {fid:3d}  trad: {[CLASS_NAMES[i][:2] for i in rt.class_ids]}')
    print(f'          ai  : {[CLASS_NAMES[i][:2] for i in ra.class_ids]}')
    print(f'          vflg: {ra.valid_flags.round(3)}')
