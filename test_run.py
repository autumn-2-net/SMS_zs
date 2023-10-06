import pytest
from utils.config_loader import get_config

def test_cfg():
    acfg=get_config('testcfg/a.yaml')
    assert acfg.get('base_config') is None
    assert acfg == {'a':1,'b':1,'c':1,'d':1}
def test_cfgb():
    acfg=get_config('testcfg/test_cfg.yaml')
    assert acfg.get('base_config') is None
    assert acfg == {'a':2,'b':1,'c':1,'d':1,'e':2}
def test_cfgbx():
    acfg=get_config('testcfg/test_cfg_base.yaml')
    assert acfg.get('base_config') is None
    assert acfg == {'a': 5, 'b': 1, 'c': 1, 'd': 1, 'e': 3, 'z': 1}
