from __future__ import absolute_import, division, unicode_literals
import argparse

from .commands.auto import Auto
from .tuner.tuner import Tuner
from .deployment.deploy import Deploy


def main():
    parser = argparse.ArgumentParser(
        description='Studio, a code-free Machine Learning toolbox for data management, training and evaluation.'
    )
    parser.add_argument('config_file')
    parser.add_argument('--tuner', dest='tuner', action='store_true', help="Use the tuner")
    parser.add_argument('--deploy', dest='deploy', action='store_true', help="Use Deploy class options")

    try:
        args = parser.parse_args()
        if args.tuner:
            tuner = Tuner(args.config_file)
            tuner.run()
        elif args.deploy:
            deploy = Deploy(args.config_file)
            deploy.run()
        else:
            auto = Auto(args.config_file)
            auto.run()

    except Exception as e:
        message = 'an unexpected error occurred: {}: {}'.format(
            type(e).__name__,
            (e.message if hasattr(e, 'message') else '') or str(e)
        )
        raise ValueError(message)


if __name__ == '__main__':
    main()
