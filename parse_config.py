from utils import read_json, write_json
from logger import setup_logging

import os
import logging
from functools import reduce
from operator import getitem
from datetime import datetime
from pathlib import Path

class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        # load config file and apply modification
        self._config = _update_config(config, modification)

        if resume is not None and self._config['resume_dir'] is not None:
            self.resume = os.path.join(self._config['resume_dir'], resume)
        else:
            self.resume = resume

        # set save_dir for ttrained model and log
        save_dir = Path(self.config['trainer']['save_dir'])
        exper_name = self.config['name']

        if run_id is None: # default id is timestamp
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')

        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id

        # make directory for saving checkpoints and logs
        exist_ok = run_id = ''
        if not os.path.exists(self._save_dir):
            self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        if not os.path.exists(self._log_dir):
            self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        # configure loggine module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def init_obj(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs) # only new kwargs allowed

        return getattr(module, module_name)(*args, **module_args)


    @classmethod
    def from_args(cls, parser, options=''):
        # Add Custom Arguments
        for opt in options:
            parser.add_argument(*opt.flags, default=None, type=opt.type)

        if not isinstance(parser, tuple):
            args = parser.parse_args()

        if args.device is not None:
            pass
        if args.resume is not None:
            resume = Path(args.resume)
            print("resume: {}".format(resume))
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Add -c config.json"
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)

        config = read_json(cfg_fname)

        # Resume
        if args.config and resume:
            config.update(read_json(args.config))

        # custom cli options into dict
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        print(modification)
        if 'run_id' in modification and modification['run_id'] is not None:
            config['run_id'] = modification['run_id']
        return cls(config, resume, modification, config['run_id'])

    def get_logger(self, name, verbosity=2):
        msg_verbosity = "verbosity option {} is invalid".format(verbosity)
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def __getitem__(self, name):
        return self.config[name]

    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir


# Helper Functions - not belong to Class
def _update_config(config, modification):
    # modification={'optimizer;args;lr': 0.01, 'data_loader;args;batch_size': None} -> modify config
    if modification is None:
        return config

    for key, val in modification.items():
        if val is not None:
            key = key.split(';')
            reduce(getitem, key[:-1], config)[key[-1]] = val

    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

