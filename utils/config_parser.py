import yaml
import os
import logging
import importlib
from ast import literal_eval


def parse_config(yaml_file_path, arg_list=None):
    """
    Load a YAML configuration file. Return an EasyDict object.
    :type yaml_file_path: str
    Examples:
    >>> config = parse_config("../experiments/default.yaml")
    >>> config.misc.log_dir = 'logs'

    # >>> #_ = config.misc.log_dir
    """
    # Read YAML experiment definition file
    with open(yaml_file_path, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)
    if arg_list:
        cfg = update_config(cfg, arg_list)
    cfg = make_paths_absolute('.', cfg)
    cfg = EasyDict(cfg)
    return cfg


class EasyDict(dict):
    """
    Copied from EasyDict
    EasyDict allows to access dict values as attributes (works recursively). A Javascript-like properties dot notation for python dicts.
    Examples:
    >>> d = EasyDict({'foo':3, 'bar':{'x':1, 'y':2}})
    >>> d.foo
    3
    >>> d.bar.x
    1

    >>> d = EasyDict(foo=3)
    >>> d.foo
    3
    """

    def __init__(self, d=None, **kwargs):
        super().__init__()
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)


def make_paths_absolute(dir_, cfg):
    """
    Make all values for keys ending with `_path` absolute to dir_.
    """
    for key in cfg.keys():
        if key.endswith("_path") or key.endswith("_dir"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.exists(cfg[key]):
                logging.warning("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg


def update_config(cfg_dict, arg_list):
    """
    Examples:
    >>> config = parse_config("../experiments/default.yaml")
    >>> config = update_config(config, ["train.batch_size", "11", "misc.log_dir", "lllogs"])

    :param config:
    :param arg_list:
    :return:
    """
    assert len(arg_list) % 2 == 0, "args should be a key-value list with even length, e.g. [k1, v1, k2, v2, ...]."
    for k, v in zip(arg_list[0::2], arg_list[1::2]):
        d = cfg_dict
        for k_ in k.split('.')[:-1]:
            d = d[k_]
        d[k.split('.')[-1]] = _decode_value(v)
    return cfg_dict


def _decode_value(value):
    """
    decode a string to python object.
    Examples:
    >>> _decode_value('./logs/xxx')
    './logs/xxx'
    >>> _decode_value('[1,2,3]')
    [1, 2, 3]
    """
    # All remaining processing is only applied to strings
    if not isinstance(value, str):
        return value
    # Try to interpret `value` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        value = literal_eval(value)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return value


if __name__ == '__main__':
    import doctest

    doctest.testmod()
