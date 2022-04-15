import logging
import time
from pathlib import Path


def create_log_dir(output_dir, conf_name, run_name=None):
    output_dir = Path(output_dir)
    # set up logger
    if not output_dir.exists():
        logging.info('=> creating {}'.format(output_dir))
        output_dir.mkdir()

    if not run_name:
        run_name = time.strftime('%m_%d_%H_%M')
    else:
        run_name = str(run_name)

    output_dir = output_dir / conf_name / run_name

    logging.info('=> creating {}'.format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_log_dir = output_dir / 'tblog'
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / 'checkpoint'
    print('=> creating {}'.format(checkpoint_dir))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir), str(tensorboard_log_dir), str(checkpoint_dir)
